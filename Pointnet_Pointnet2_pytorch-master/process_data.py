# 读取文件内容
with open("test_part_seg.txt", "r") as file:
    lines = file.readlines()

# 处理每一行数据
new_lines = []
for line in lines:
    # 移除行末尾的换行符，并用空格替换逗号
    new_line = line.strip().replace(",", " ")
    # 添加额外的0
    new_line += " 0\n"
    new_lines.append(new_line)

# 将处理后的数据写入新文件
with open("processed_test_part_seg.txt", "w") as file:
    file.writelines(new_lines)

print("处理完成！")
