// 第1天：PCL hello world
// 生成一片随机点云，用 VoxelGrid 体素降采样，打印点数和耗时。
#include <chrono>
#include <iostream>
#include <random>

#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

int main() {
    // 1. 生成 10 万个点：12m×12m 的平地，高度带 2cm 噪声（和第1课的草地类似）
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> xy(-6.0f, 6.0f);
    std::normal_distribution<float> z(0.0f, 0.02f);
    for (int i = 0; i < 100000; ++i) {
        cloud->push_back(pcl::PointXYZ(xy(rng), xy(rng), z(rng)));
    }

    // 2. 体素降采样：5cm 格子
    pcl::PointCloud<pcl::PointXYZ>::Ptr down(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setInputCloud(cloud);
    voxel.setLeafSize(0.05f, 0.05f, 0.05f);

    auto t0 = std::chrono::steady_clock::now();
    voxel.filter(*down);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "原始点数 " << cloud->size() << " → 降采样后 " << down->size()
              << "，耗时 " << ms << " ms" << std::endl;
    return 0;
}
