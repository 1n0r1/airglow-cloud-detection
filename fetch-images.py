import os

repo_ASI = "data"

year = 2022

for site in ['blo']:
    for doy in range(1, 366):
        cmd = '/usr/bin/wget -r -nH -nd --cut-dirs=8 -A png --no-parent -P %s/%s/%s https://data.mangonetwork.org/data/transport/mango/archive/%s/greenline/raw/%d/%03d/ &' % (repo_ASI, site, year, site, year, doy)
        file = open('cmd-list.sh', 'a+')
        file.write(cmd + '\n')
        print(cmd)