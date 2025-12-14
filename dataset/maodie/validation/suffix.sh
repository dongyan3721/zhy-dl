#!/bin/bash

# 遍历当前目录下的所有.png文件
for file in *.png; do
    # 检查文件是否存在（避免无.png文件时的错误）
    if [ -f "$file" ]; then
        # 获取文件名（不含扩展名）
        filename="${file%.*}"
        # 获取扩展名（包含点）
        extension=".${file##*.}"
        # 新文件名
        newname="${filename}-1${extension}"
        # 重命名文件
        mv "$file" "$newname"
        echo "已重命名: $file -> $newname"
    fi
done

echo "所有.png文件已添加-1后缀"
