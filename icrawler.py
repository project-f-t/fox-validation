from icrawler.builtin import BingImageCrawler
import random
import shutil
import glob
import os

#クローリング定義
def crawling(keyword, num): #検索キーワード、最大取得枚数

  #画像保存するdirを生成
  path_train = "/content/drive/MyDrive/Code/Fox Verification/dataset/train/" + keyword
  os.makedirs(path_train,exist_ok=True)
  path_test = "/content/drive/MyDrive/Code/Fox Verification/dataset/validation/" + keyword
  os.makedirs(path_test,exist_ok=True)

  #クローラーの設定
  crawler = BingImageCrawler(downloader_threads = 4, storage={"root_dir":path_train})
  crawler.crawl(keyword=keyword, max_num = num)

  #取得したデータのリスト
  img_lst = glob.glob(path_train + "/*")

  #データリストからテストデータと検証データに分類
  for img in img_lst:
    p = random.random()
    if p <= 0.2:
      try:
        shutil.move(img, path_test+"/")
      except Exception as e:
        print(e.args)
