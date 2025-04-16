import sys
t = int(input())
result = []
value_list = []
key_dict = {}

line = sys.stdin.readline().strip()
inputs = list(line.split())

if len(inputs) != t:
    exit(0)

for i in inputs:
    value = ord(i[0])
    value_list.append(value)

key_dict.update(dict(zip(inputs, value_list)))
key_dict = dict(sorted(key_dict.items()))

for key in key_dict.keys():
    result.append(key)
for i in range(len(result)):
    print(result[i],end=' ')
