// 点云预处理流程（计划第 2–3 天）
//
//   读 KITTI .bin → 范围裁剪 → 体素降采样 → 统计离群点去除
//   → RANSAC 去地面 → 欧式聚类 → 输出每步点数和耗时
//
// 用法：
//   ./preprocess <一个.bin文件 或 一个装满.bin的目录> [结果csv路径]
// 例：
//   ./preprocess ~/data/kitti/sequences/08/velodyne results/preprocess_08.csv
//
// KITTI .bin 格式：每个点 4 个 float32（x, y, z, 反射强度），小端存储。
// KITTI 雷达离地约 1.73m，所以地面大约在 z = -1.73 附近。

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <pcl/ModelCoefficients.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>

namespace fs = std::filesystem;
using PointT = pcl::PointXYZI;
using Cloud = pcl::PointCloud<PointT>;

// ---------------- 参数：改这里做实验，并把改动记进 README ----------------
struct Params {
    Eigen::Vector4f crop_min{-20.f, -20.f, -3.f, 1.f};
    Eigen::Vector4f crop_max{20.f, 20.f, 3.f, 1.f};
    float voxel_leaf = 0.10f;       // 体素边长 (m)
    int sor_k = 16;                 // SOR 近邻数
    double sor_std = 1.0;           // SOR 标准差倍数
    double ground_dist = 0.15;      // RANSAC 地面距离阈值 (m)
    int ransac_iters = 200;
    double cluster_tol = 0.30;      // 聚类连接距离 (m)，要明显大于 voxel_leaf
    int cluster_min = 20;
    int cluster_max = 20000;
};

Cloud::Ptr load_kitti_bin(const std::string& path) {
    Cloud::Ptr cloud(new Cloud);
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "打不开文件: " << path << std::endl;
        return cloud;
    }
    float buf[4];
    while (f.read(reinterpret_cast<char*>(buf), sizeof(buf))) {
        PointT p;
        p.x = buf[0]; p.y = buf[1]; p.z = buf[2]; p.intensity = buf[3];
        cloud->push_back(p);
    }
    return cloud;
}

class Timer {
public:
    Timer() : t0_(std::chrono::steady_clock::now()) {}
    double ms() const {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_).count();
    }
private:
    std::chrono::steady_clock::time_point t0_;
};

struct FrameResult {
    std::string name;
    size_t n_raw = 0, n_crop = 0, n_voxel = 0, n_sor = 0, n_ground = 0, n_obstacle = 0;
    size_t n_clusters = 0;
    double t_crop = 0, t_voxel = 0, t_sor = 0, t_ground = 0, t_cluster = 0;
};

FrameResult process(const std::string& path, const Params& P) {
    FrameResult r;
    r.name = fs::path(path).filename().string();
    Cloud::Ptr raw = load_kitti_bin(path);
    r.n_raw = raw->size();

    // 1. 范围裁剪
    Cloud::Ptr crop(new Cloud);
    {
        Timer t;
        pcl::CropBox<PointT> box;
        box.setMin(P.crop_min);
        box.setMax(P.crop_max);
        box.setInputCloud(raw);
        box.filter(*crop);
        r.t_crop = t.ms();
    }
    r.n_crop = crop->size();

    // 2. 体素降采样
    Cloud::Ptr voxel(new Cloud);
    {
        Timer t;
        pcl::VoxelGrid<PointT> vg;
        vg.setInputCloud(crop);
        vg.setLeafSize(P.voxel_leaf, P.voxel_leaf, P.voxel_leaf);
        vg.filter(*voxel);
        r.t_voxel = t.ms();
    }
    r.n_voxel = voxel->size();

    // 3. 统计离群点去除
    Cloud::Ptr clean(new Cloud);
    {
        Timer t;
        pcl::StatisticalOutlierRemoval<PointT> sor;
        sor.setInputCloud(voxel);
        sor.setMeanK(P.sor_k);
        sor.setStddevMulThresh(P.sor_std);
        sor.filter(*clean);
        r.t_sor = t.ms();
    }
    r.n_sor = clean->size();

    // 4. RANSAC 单平面去地面（基线方法；Patchwork++ 的对比在 scripts/eval_ground.py）
    Cloud::Ptr obstacle(new Cloud);
    {
        Timer t;
        pcl::SACSegmentation<PointT> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);   // 只接受接近水平的平面
        seg.setAxis(Eigen::Vector3f(0.f, 0.f, 1.f));
        seg.setEpsAngle(15.0 * M_PI / 180.0);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(P.ground_dist);
        seg.setMaxIterations(P.ransac_iters);
        seg.setInputCloud(clean);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients coeff;
        seg.segment(*inliers, coeff);

        pcl::ExtractIndices<PointT> ex;
        ex.setInputCloud(clean);
        ex.setIndices(inliers);
        ex.setNegative(true);            // 取“不是地面”的点
        ex.filter(*obstacle);
        r.n_ground = inliers->indices.size();
        r.t_ground = t.ms();
    }
    r.n_obstacle = obstacle->size();

    // 5. 欧式聚类
    {
        Timer t;
        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
        tree->setInputCloud(obstacle);
        std::vector<pcl::PointIndices> clusters;
        pcl::EuclideanClusterExtraction<PointT> ec;
        ec.setClusterTolerance(P.cluster_tol);
        ec.setMinClusterSize(P.cluster_min);
        ec.setMaxClusterSize(P.cluster_max);
        ec.setSearchMethod(tree);
        ec.setInputCloud(obstacle);
        ec.extract(clusters);
        r.n_clusters = clusters.size();
        r.t_cluster = t.ms();
    }
    return r;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <.bin文件或目录> [结果.csv]" << std::endl;
        return 1;
    }
    std::vector<std::string> files;
    fs::path in(argv[1]);
    if (fs::is_directory(in)) {
        for (auto& e : fs::directory_iterator(in))
            if (e.path().extension() == ".bin") files.push_back(e.path().string());
        std::sort(files.begin(), files.end());
    } else {
        files.push_back(in.string());
    }
    if (files.empty()) {
        std::cerr << "没有找到 .bin 文件" << std::endl;
        return 1;
    }

    Params P;
    std::ofstream csv;
    if (argc >= 3) {
        csv.open(argv[2]);
        csv << "frame,n_raw,n_crop,n_voxel,n_sor,n_ground,n_obstacle,n_clusters,"
               "t_crop_ms,t_voxel_ms,t_sor_ms,t_ground_ms,t_cluster_ms\n";
    }

    double sum[5] = {0, 0, 0, 0, 0};
    for (const auto& f : files) {
        FrameResult r = process(f, P);
        sum[0] += r.t_crop; sum[1] += r.t_voxel; sum[2] += r.t_sor;
        sum[3] += r.t_ground; sum[4] += r.t_cluster;
        std::cout << r.name << "  点数 " << r.n_raw << " → 裁剪 " << r.n_crop << " → 体素 " << r.n_voxel
                  << " → SOR " << r.n_sor << " | 地面 " << r.n_ground << " 障碍 " << r.n_obstacle
                  << " | 簇 " << r.n_clusters << std::endl;
        if (csv) {
            csv << r.name << ',' << r.n_raw << ',' << r.n_crop << ',' << r.n_voxel << ',' << r.n_sor << ','
                << r.n_ground << ',' << r.n_obstacle << ',' << r.n_clusters << ',' << r.t_crop << ','
                << r.t_voxel << ',' << r.t_sor << ',' << r.t_ground << ',' << r.t_cluster << '\n';
        }
    }
    const char* names[5] = {"裁剪", "体素", "SOR", "地面", "聚类"};
    std::cout << "\n平均耗时（" << files.size() << " 帧）:";
    for (int i = 0; i < 5; ++i) std::cout << "  " << names[i] << " " << sum[i] / files.size() << "ms";
    std::cout << std::endl;
    return 0;
}
