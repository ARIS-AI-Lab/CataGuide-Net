import xml.etree.ElementTree as ET

# 加载 XML 文件
tree = ET.parse(r'C:\Users\Charl\Downloads\CaDTD-main\CaDTD-main\Setup 2\VOC Labels\Video10_frame019550.xml')
root = tree.getroot()

# 获取根节点
filename = root.find('filename').text
print(f"Filename: {filename}")

# 获取图像的尺寸信息
size = root.find('size')
width = size.find('width').text
height = size.find('height').text
depth = size.find('depth').text
print(f"Image Size - Width: {width}, Height: {height}, Depth: {depth}")

# 遍历所有的 object 标签
for obj in root.findall('object'):
    name = obj.find('name').text
    pose = obj.find('pose').text
    truncated = obj.find('truncated').text
    difficult = obj.find('difficult').text

    # 获取边界框信息
    bndbox = obj.find('bndbox')
    xmin = bndbox.find('xmin').text
    ymin = bndbox.find('ymin').text
    xmax = bndbox.find('xmax').text
    ymax = bndbox.find('ymax').text

    # 输出 object 信息
    print(f"\nObject: {name}")
    print(f"  Pose: {pose}")
    print(f"  Truncated: {truncated}")
    print(f"  Difficult: {difficult}")
    print(f"  Bounding Box: ({xmin}, {ymin}, {xmax}, {ymax})")