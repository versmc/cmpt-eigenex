
# CMakeのバージョンを設定
#cmake_minimum_required(VERSION 2.8)

# プロジェクト名と使用する言語を設定
project(hello CXX)
# hello という実行ファイルをmain.cppから作成
add_executable(hello hello.cpp)
