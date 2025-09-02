# 测试微博是否登录成功
import requests

cookies = {
    "SUB": "_2A25FsivMDeRhGedG41sQ9SnIyj-IHXVmziEErDV8PUNbmtANLUaskW9NUOk2bX3oBwt9lpeoTCjkFIufzwVlYmyA",
    "SUBP": "0033WrSXqPxfM725Ws9jqgMF55529P9D9WFj7cf7HiukpvQikGSgy-5z5NHD95Qp1hn4eK-NSh20Ws4DqcjMi--NiK.Xi-2Ri--ciKnRi-zN1KLDMN90UNeN15tt",
}

url = "https://weibo.com/"
res = requests.get(url, cookies=cookies)
print(res.status_code)
print(res.text[:500])  # 打印前500个字符
