所需环境在requirements文档中
训练请运行main.py,并将第112行和114行改为自己的标签数据路径和图片数据路径
测试模型请运行test_code,并将第42行和44行改为自己的标签数据路径和图片数据路径,第153行改为自己的预训练模型路径
如果想获取分类错误图片，请在test_code第227行将showerros改为true，并修改第92行为存储错误图片的路径。
分类错误图片示例在error文件夹里。