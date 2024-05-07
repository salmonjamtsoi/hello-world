

import pandas as pd
import numpy as np
import requests
import re
from math import gcd
from rdkit import Chem
from collections import Counter



def divide_by_gcd(string):
    # 提取字符串中的数字部分
    numbers = re.findall(r'\d+', string)

    if not numbers:
        return string

    # 计算数字列表中的最大公约数
    if len(numbers) == 1:
        max_gcd = int(numbers[0])
    else:
        max_gcd = gcd(*map(int, numbers))

    # 将每个数字除以最大公约数，并构建新的字符串
    new_string = re.sub(r'\d+', lambda m: str(int(m.group()) // max_gcd), string)

    return new_string



def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return True
    return False



def API2Gad(molecule,material,T):

    mol = Chem.MolFromSmiles(molecule)
    if mol is not None:
        url = "http://10.30.73.198:31702/api/molecules/page/"
        data = {
            "data": {
            },
            "requestBase": {
                "page": "0-0",
                "sort": "-createTime"
            }
        }
        try:
            response = requests.post(url,json=data)
            result = response.json().get('content').get('records')

        except requests.exceptions.RequestException as e:
            print("请求错误:", e)

        idA = ''
        for i in result:
            smi = i.get('smiles')
            if smi == molecule:
                idA = i.get('idA')
                E_ref = i.get('eref')

        if idA:
            # 计算分子包含的元素种类及数量
            mol = Chem.MolFromSmiles(molecule)
            mol = Chem.AddHs(mol)
            # 获取原子列表
            atoms = mol.GetAtoms()
            # 计算原子种类及其数量
            atom_counts = Counter([atom.GetSymbol() for atom in atoms])

            G_cor = []

            for symbol, count in atom_counts.items():
                url = "http://10.30.73.198:31702/api/element/page/"
                data = {
                    "data": {
                    },
                    "requestBase": {
                        "page": "0-0",
                        "sort": "-createTime"
                    }
                }
                try:
                    response = requests.post(url,json=data)
                    result = response.json().get('content').get('records')

                except requests.exceptions.RequestException as e:
                    print("请求错误:", e)
                for i in result:
                    elename = i.get('eleName')
                    if elename == symbol:
                        a_ele = i.get('a')
                        b_ele = i.get('b')
                        c_ele = i.get('c')
                        G_cor_ele = a_ele*(T**2) + b_ele*T + c_ele

                        G_cor_ele = count*G_cor_ele
                        G_cor.append(G_cor_ele)
            G_cor_ref = sum(G_cor)
            G_ref = E_ref + G_cor_ref

            G_ad_list = []

            url = "http://10.30.73.198:31702/api/point/page/"
            data = {
                "data": {
                },
                "requestBase": {
                    "page": "0-0",
                    "sort": "-createTime"
                }
            }
            try:
                response = requests.post(url,json=data)
                result = response.json().get('content').get('records')
                for i in result:
                    ida = i.get('idA')
                    mate = i.get('idM').split('-')[0]
                    new_mate = divide_by_gcd(mate)

                    if ida == idA and new_mate == material :
                        E_ads = i.get('etotalVasp')
                        a_ads = i.get('a')
                        b_ads = i.get('b')
                        c_ads = i.get('c')
                        E_cat = i.get('etotalVaspM')

                        G_ads = E_ads + a_ads*T**2 + b_ads*T + c_ads
                        G_ad = G_ads - G_ref - E_cat

                        G_ad_list.append(G_ad)

            except requests.exceptions.RequestException as e:
                print("请求错误:", e)

            if G_ad_list:
                G_ad_list = min(G_ad_list)
                return G_ad_list
            else:
                print('Not Found')

        else:
            print('数据库中不存在该分子:',molecule)
            return None

    else:
        print(molecule,'该结构不存在')
        return None




def fitting(x1_list,x2_list,y_list):
    A = np.vstack((x1_list, x2_list, np.ones(len(x1_list)))).T
    b = np.array(y_list)

    # 使用最小二乘法求解参数
    params = np.linalg.lstsq(A, b, rcond=None)[0]

    # 提取参数
    a = params[0]
    b = params[1]
    c = params[2]

    y_pred = a * np.array(x1_list) + b * np.array(x2_list) + c

    # 计算SST
    y_mean = np.mean(y_list)
    SST = np.sum((y_list - y_mean) ** 2)

    # 计算SSE
    SSE = np.sum((y_list - y_pred) ** 2)

    # 计算R2
    R2 = 1 - SSE / SST

    return a, b, c, R2



def Descriptor(data):

    index_list = list(range(data.shape[1]))
    index_com = []
    for i in range(len(index_list)):
        for j in range(i+1, len(index_list)):
            combination = (index_list[i], index_list[j])
            index_com.append(combination)

    R2_all_list = []
    index_record_list = []
    for i in index_com:
        x_index = list(i)
        y_index = [i for i in index_list if i not in x_index]
        R2_list = []
        for k in y_index:
            x1_lis = []
            x2_lis = []
            y_lis = []
            for j in range(data.shape[0]):
                x1 = data.iloc[j,x_index[0]]
                x2 = data.iloc[j,x_index[1]]
                y = data.iloc[j,k]
                x1_lis.append(x1)
                x2_lis.append(x2)
                y_lis.append(y)
            a, b, c, R2 = fitting(x1_lis,x2_lis,y_lis)
            R2_list.append(R2)
            # print('=============')
        R2_ave = sum(R2_list) / len(R2_list)
        R2_var = np.var(R2_list)
        R2_max = max(R2_list)
        R2_min = min(R2_list)
        R2_med = np.median(R2_list)
        R2_all = 0.6*R2_ave + -0.1*R2_var + 0.1*R2_max + 0.1*R2_min + 0.1*R2_med
        R2_all_list.append(R2_all)

        index_record_list.append(i)


    max_r2 = max(R2_all_list)
    max_index = R2_all_list.index(max_r2)

    id = index_record_list[max_index]

    descrip = [data.columns[i] for i in id]

    return descrip



molecule = ['[H]','C[CH]C','CC[CH2]','CC[CH]','CC=C','[CH2]C[CH2]','C[C]C']
material = ['Pt1', 'Pt7Sn1', 'Pt3Sn1', 'Pt5Sn3']
T = 873



updated_material = [divide_by_gcd(item) for item in material]



# 创建一个空的 DataFrame 来存储计算结果
data = pd.DataFrame(index=updated_material, columns=molecule)



# 对每个元素进行计算并填充到 DataFrame 中
for m in molecule:
    for mat in material:
        result = API2Gad(m, mat, T)
        data.at[mat, m] = result
print(data)


null_values = data.isnull()
if null_values.any().any():
    print("数据未知，请检查")
else:
    descrip = Descriptor(data)
    print(descrip)
