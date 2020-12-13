import os
import sys


# PROJECT_DIR = '/home/uoneway/Project/PreSumm_ko'
PROJECT_DIR = '../..'

os.chdir(PROJECT_DIR)
os.makedirs('temp', exist_ok=True)
os.chdir('temp')

# mecab 설치
print("-----------------------mecab 설치--------------------------")
os.system("""
curl -LO https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
tar zxfv mecab-0.996-ko-0.9.2.tar.gz
cd mecab-0.996-ko-0.9.2
./configure
make
make check
make install
ldconfig
""")

print("-----------------------GNU M4 설치--------------------------")
os.system("""
wget http://ftp.gnu.org/gnu/m4/m4-1.4.9.tar.gz
tar -zvxf m4-1.4.9.tar.gz
cd m4-1.4.9
./configure
make
make install
""")

print("-----------------------autoconf 설치--------------------------")
os.system("""
curl -OL http://ftpmirror.gnu.org/autoconf/autoconf-2.69.tar.gz
tar xzf autoconf-2.69.tar.gz
cd autoconf-2.69
./configure --prefix=/usr/local
make
make install
export PATH=/usr/local/bin
""")

print("-----------------------automake 설치--------------------------")
os.system("""
curl -LO http://ftpmirror.gnu.org/automake/automake-1.11.tar.gz
tar -zxvf automake-1.11.tar.gz
cd automake-1.11
./configure
make
make install
""")

print("-----------------------mecab dictionary 설치--------------------------")
os.system("""
curl -LO https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
tar -zxvf mecab-ko-dic-2.1.1-20180720.tar.gz
cd mecab-ko-dic-2.1.1-20180720
./autogen.sh
./configure
make
make install
""")

print("-----------------------mecab python 설치--------------------------")
os.system("""
git clone https://bitbucket.org/eunjeon/mecab-python-0.996.git
cd mecab-python-0.996
python3 setup.py build
python3 setup.py install
""")

# pip install mecab-python3