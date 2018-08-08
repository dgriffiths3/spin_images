#include <iostream>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/spin_image.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/histogram_visualizer.h>

typedef pcl::Histogram<153> SpinImage;

void visualize_points(const pcl::PointCloud<pcl::PointXYZ>::Ptr points,
               const pcl::PointCloud<pcl::Normal>::Ptr normals)
               {
                 pcl::visualization::PCLVisualizer viz;
                 viz.addPointCloud(points, "points");
                 viz.addPointCloudNormals<pcl::PointXYZ, pcl::Normal> (points, normals, 10, 0.1, "normals");
                 viz.spin();
               }

int main()
{
  // Load in point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ> ("/home/david/data/spin_images/cpp/random_pc.pcd", *cloud) == -1)
  {
    PCL_ERROR("Could not read file");
    return -1;
  }

  // Compute normals for point cloud
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimation;
  normal_estimation.setInputCloud (cloud);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree (new pcl::search::KdTree<pcl::PointXYZ>);
  normal_estimation.setSearchMethod (kdtree);
  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud< pcl::Normal>);
  normal_estimation.setRadiusSearch (0.3);
  normal_estimation.compute (*normals);

  //visualize_points(cloud, normals);

  // Compute Spin Images

  pcl::PointCloud<SpinImage>::Ptr spin_images(new pcl::PointCloud<SpinImage>());
  pcl::SpinImageEstimation<pcl::PointXYZ, pcl::Normal, SpinImage> si(8, 0.5, 8);
  si.setInputCloud(cloud);
  si.setInputNormals(normals);
  si.setRadiusSearch(1);
  si.compute(*spin_images);

  std::cout << spin_images->points[0] << std::endl;


  //std::cout << "SI output points size: " << spin_images->points.size() << std::endl;
  //std::cout << spin_images->points[0] << std::endl;

  return 0;
}
