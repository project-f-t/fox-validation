# fox-validation
[For my study] image recognition test. verifing JP red fox, Tibetan fox, and Yoshioka Riho Dongitune. 

内容
####
日本キツネとチベットスナギツネ、およびどんぎつね（吉岡里帆様）を認識・識別する画像識別モデルの生成を通して、
データの収集方法、データ処理、Fine-Tuning手法とモデルトレーニング(keras)を学ぶ、初学者による初学者のための戯れ
####

今回は、Google ColabのGPUを使って学習を進めた。

1. icrawler.pyによって、データを収集している。
注意！このコードを実行する前に、Google ColabにGoogleDriveをマウントする必要がある。
icrawlerはたったの数行でネット上から画像データを大量に収集することが出来る。
画像認識の勉強のためには、最も工数のかかるデータの収集を簡便にできるため、とてもありがたいライブラリ。
icrawler.builtinには、Bingのほかgoogle,Baiduがあるが、Google検索はどうやらエラーが出る模様。。。

2. model.pyで実際のモデル生成およびトレーニング
今回のモデルは、一から生成するのではなく、VGG16を利用したFine-Tuningという手法を使用する。
VGG16の各層の学習を凍結した状態で、FC層（FlattenからSoftmaxまでの最後の層）だけをトレーニングする。
今回の手法によって、精度を上げるために必要となる膨大なモデルの生成及び学習を行わず、限られたリソースの中でも圧倒的な制度を実現できることが出来るとわかった。
Fine-Tuningはどうやら教義的な意味の転移学習と言えるらしいが、転移学習恐るべしと感じた。転移学習の波が来ている理由が分かった気がする。
また、このソースコードではkerasのImageDataGeneratorの便利さを知った。画像データを 1.サイズを画一化　2.テンソル化 という工程をGeneratorを使えば瞬時に終わらせることが出来る。
そしてそのままmodel.fit(generator)とすれば学習が進む。ちなみに、fit_generator()もあるが、fit()がgeneratorに対応したため、区別する必要がないようだ。
また、Generatorを生成する際に、指定するディレクトリが
・dataset-train-class1,class2,...
と各クラスのディレクトリを分けて保存していれば、自動でクラス数を認識してくれるそうだ。
--> train_generator.class_indices
