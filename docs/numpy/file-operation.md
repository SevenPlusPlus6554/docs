
# NumPy 数据文件的读写操作
## 读写二进制文件“.npy”
NumPy 的 save()函数将数组以二进制格式保存在扩展名为 .npy 的文件中，可以使用 load() 函数读取 .npy 的文件中的数据。语法格式为：

numpy.save(file, arr)

file：用来保存数组的文件名，是字符串类型。

arr：要保存的数组。
## 读写二进制文件“. npz”
savez() 函数用于将多个数组写入同一个扩展名为 .npz 的文件中，同样使用 load() 函数读取。语法格式为：

numpy.savez(file, *args, **kwds)

arrs = numpy.load(file)

file, 用来保存数组的文件名，是字符串类型。

*args，非关键字数组名，自动按顺序取名。

**kwds，关键字数组名，按关键字取名。
## 读写文本文件（只能存取1维或2维数组）
savetxt() 函数将数组写入文本文件 (.txt，.csv 等)，语法格式为：

numpy.savetxt(file, arr, fmt="%.18e", delimiter=' ', newline='\n' )

file： 用来保存数组的文本文件名

arr： 要保存的数组。

fmt：指定数据存入的格式。

delimiter：数据列之间的分隔符，数据类型为字符串，默认值为：' '。

newline ： 数据行之间的分隔符。

loadtxt() 函数读取文本文件中的数据，加载数据的语法格式：

arr = numpy.loadtxt(file, dtype=int, delimiter=' ')