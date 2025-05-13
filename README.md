# GeoAlignPy
GeoAlignPy: 高精度遥感影像配准工具，结合地理坐标系保持与ECC算法，实现精确像素级配准。

功能特点

🌏 保留地理参考 - 在配准过程中正确处理和保存地理坐标信息

🔍 亚像素级精度 - 采用ECC算法实现高精度配准

🔄 自动重投影 - 自动处理不同坐标系间的转换

📊 可视化比较 - 提供配准前后对比可视化

🛠️ 易于使用 - 简单的API，适合遥感工作流集成


环境要求

Python 3.7+

rasterio

numpy

opencv-python

matplotlib


技术原理
GeoAlignPy结合了两个关键步骤来确保精确配准:

空间重投影 - 使用rasterio将待配准影像重投影到参考影像的坐标系统

ECC精配准 - 使用OpenCV的ECC算法进行像素级精确配准

地理坐标保持 - 通过数学变换整合像素配准与地理坐标系统


这种方法确保了配准后影像既有像素级的精确对齐，又保留了正确的地理坐标信息。

应用场景

多时相遥感影像变化检测

多源遥感数据融合

时序遥感数据分析

卫星影像镶嵌

精确制图与测量


代码中修改路径：
    
    ref_img_path = r"H:\1CD_dataset\001\GF2_PMS2_E114.8_N22.8_20231125_L1A13495975001\GF2_PMS2_E114.8_N22.8_20231125_L1A13495975001-MSS2.tiff"
    
    mov_img_path = r"H:\0ZJU_CD\jz\GF2_PMS2_E114.8_N22.8_20220403_L1A0006387395\GF2_PMS2_E114.8_N22.8_20220403_L1A0006387395-MSS2.tiff"
    
    output_path = "aligned_result.tif"
    
引用
如果GeoAlignPy对您的研究有帮助，请考虑引用:

@software{geoalignpy2025,
  author = {Xiunan Li},
  title = {GeoAlignPy: A Python Tool for High-Precision Remote Sensing Image Registration with Geographic Reference Preservation},
  year = {2025},
  url = {https://github.com/smartRemoteSensing/GeoAlignPy}
}
