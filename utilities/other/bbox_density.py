import os

import matplotlib.pyplot as plt
import numpy as np
from bbox_distribution import BboxHistogram
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA


class BboxDensity:
    def bbox_density_heatmap(self, path, save_path):

        bh = BboxHistogram(args.supported_file_keys, args.path)
        resolutions = bh.image_resolution_by_folder(path)
        bbox_histograms,bbox_center_image = bh.bbox_resolution(resolutions)

        for folder in bbox_center_image.keys():

            savepath = save_path
            folderpath = folder.split(os.sep)

            for n in range(3, len(folderpath)):
                if not os.path.exists(savepath + os.sep + folderpath[n]):
                    os.makedirs(savepath + os.sep + folderpath[n])
                savepath = savepath + os.sep + folderpath[n]

            for key, value in bbox_center_image[folder].items():

                item_name = key.split(os.sep)[-1]
                if folder not in key:
                    break
                if not value[0]:
                    continue
                #這個問題還沒解決
                if len(value[0]) < 3:
                    continue

                #pca = PCA(n_components=2)
                point_list = np.array([[point[0], point[1]] for point in value[0]])
                #data_pca = pca.fit_transform(point_list)

                try:
                    kde = gaussian_kde(point_list.T)
                except:
                    continue

                image_width, image_height  = resolutions[folder][key]  # 图像大小（根据实际图像调整）
                x_grid, y_grid = np.meshgrid(np.linspace(0, image_width, image_width),
                                                np.linspace(0, image_height, image_height))
                grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])

                # 计算密度
                density_values = kde(grid_coords)

                # 将1维数组形状转换回2D
                density_map = density_values.reshape(image_height, image_width)

                p_name = '.'.join(item_name.split('.')[0:-1])
                # 可视化密度地图（可选）
                plt.imshow(density_map, cmap='hot', interpolation='nearest')
                plt.colorbar()
                plt.title(f"Density Map for {p_name}")
                plt.savefig(f"{savepath}/{p_name}.png")
                plt.close('all')

    def bbox_density_heatmap_by_folder(self, path, save_path):

        bh = BboxHistogram(args.supported_file_keys, args.path)
        resolutions = bh.image_resolution_by_folder(path)
        bbox_histograms,bbox_center_image = bh.bbox_resolution(resolutions)

        for folder in bbox_center_image.keys():

            density_map = []
            savepath = save_path
            folderpath = folder.split(os.sep)

            for n in range(3, len(folderpath)):
                if not os.path.exists(savepath + os.sep + folderpath[n]):
                    os.makedirs(savepath + os.sep + folderpath[n])
                savepath = savepath + os.sep + folderpath[n]

            f = open(savepath + os.sep + "density_normal.txt", "w+")

            for key, value in bbox_center_image[folder].items():

                item_name = key.split(os.sep)[-1]
                if folder not in key:
                    break
                if not value[0]:
                    continue
                #這個問題還沒解決
                #if len(value[0]) < 3:
                #    continue

                point_list = np.array([[point[0], point[1]] for point in value[0]])
                kde = gaussian_kde(point_list.T)

                image_width, image_height  = resolutions[folder][key]  # 图像大小（根据实际图像调整）
                x_grid, y_grid = np.meshgrid(np.linspace(0, image_width, image_width),
                                                np.linspace(0, image_height, image_height))
                grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])

                # 计算密度
                density_values = np.argmax(kde(grid_coords))/np.sum(kde(grid_coords))

                print(key + ":" + str(density_values),file=f )

                # 将1维数组形状转换回2D
                density_map.append(density_values)

            # 可视化密度地图（可选）
            plt.hist(density_map, bins=20, alpha = 0.5)
            plt.title(f"Density Map for {folder.split(os.sep)[-1]}")
            plt.savefig(f"{savepath}/density_normal.png")
            plt.close('all')
            f.close()

    #ratio * 數量
    def bbox_density_area_count(self, path, save_path):

        bbox_area_score = []
        bbox_key_area = {}
        bh = BboxHistogram(args.supported_file_keys, args.path)
        resolutions = bh.image_resolution_by_folder(path)
        bbox_histograms,bbox_center_image = bh.bbox_resolution(resolutions)

        for folder in bbox_center_image.keys():
            savepath = save_path
            folderpath = folder.split(os.sep)

            for n in range(3, len(folderpath)):
                if not os.path.exists(savepath + os.sep + folderpath[n]):
                    os.makedirs(savepath + os.sep + folderpath[n])
                savepath = savepath + os.sep + folderpath[n]

            f = open(savepath + os.sep + "density.txt", "w+")

            for key, value in bbox_center_image[folder].items():
                if folder not in key:
                    break
                if not value[0]:
                    continue

                #print(key + ": " + str(len(value[0]) * value[1]), file=f)
                bbox_key_area[key] = len(value[0]) * value[1]
                bbox_area_score.append(len(value[0]) * value[1])

            plt.hist(bbox_area_score, bins=20, alpha = 0.5)
            plt.xlabel("density")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(f"{savepath}/density_count.png")
            plt.close()

            for key, value in sorted(bbox_key_area.items(), key=lambda x:x[1], reverse=True):
                print(key + ": " + str(value) , file=f)

            f.close()

    #ratio
    def bbox_density_area_ratio(self, path, save_path):

        bbox_area_ratio = []
        bh = BboxHistogram(args.supported_file_keys, args.path)
        resolutions = bh.image_resolution_by_folder(path)
        bbox_histograms,bbox_center_image = bh.bbox_resolution(resolutions)

        for folder in bbox_center_image.keys():
            savepath = save_path
            folderpath = folder.split(os.sep)

            for n in range(3, len(folderpath)):
                if not os.path.exists(savepath + os.sep + folderpath[n]):
                    os.makedirs(savepath + os.sep + folderpath[n])
                savepath = savepath + os.sep + folderpath[n]

            f = open(savepath + os.sep + "density_ratio.txt", "w+")

            for key, value in sorted(bbox_center_image[folder].items(), key=lambda x: x[1][1], reverse=True):
                if folder not in key:
                    break
                if not value[0]:
                    continue

                print(key + ": " + str(value[1]), file=f)

                bbox_area_ratio.append(value[1])

            plt.hist(bbox_area_ratio, bins=20, alpha = 0.5)
            plt.xlabel("density")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(f"{savepath}/density_ratio.png")
            plt.close()

            f.close()


if __name__ == "__main__":
    import argparse

    #要有image跟label
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default=r"D:\data\translate_all\TinyObjectDataset\sku110k", help="your path")
    parser.add_argument("-sp","--savepath", type=str, default=r"D:\data\bbox_density", help="your save path")
    parser.add_argument("-sfk","--supported_file_keys", type=list, default= ['.jpg','.bmp','.png','.jpeg'], help="your supported file keys")
    args = parser.parse_args()
    bd = BboxDensity()
    #bd.bbox_density_heatmap(args.path, args.savepath)
    #bd.bbox_density_area_ratio(args.path, args.savepath)
    #bd.bbox_density_area_count(args.path, args.savepath)
    bd.bbox_density_heatmap_by_folder(args.path, args.savepath)
