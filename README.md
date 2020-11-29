### 第四周作业
#### 使用bert提取句向量或句向量（二选一），替换 BiDAF中的embedding部分，可采用以下两种方案：
- 1、直接参考bert源码中的extract_features提取特征，将bert词/句向量整理成文件，在Bi-DAF代码中的preprocess中加载，可参考glove的实现代码。
- 2、可以采用bert-as-service工具，提取bert的词/句向量信息，除采用静态文件加载外，也可以在preprocess中构造bert-as-service的client直接请求其server获取向量保存在变量中，server可在外部启动。
  
### 作业代码说明
- 1. 本作业采用第二种方式提取句向量，即采用bert-as-service工具，提取bert词/句向量信息进行静态文件加载
- 2. 采用了12层, 768个隐藏层, 12个多头, 110M 个参数的bert预训练模型BERT-Base, Uncased，并将这个预训练模型保存为tmp文件夹中的tmp/uncased_L-12_H-768_A-12，这里，通过观察文件树，可以观察到里面的vocab.txt里面包含了日文、朝鲜文、阿拉伯文、英文等多个语言的单词，但是没有简体中文内容。
- 3. 预先安装python3.6和tensorflow 1.13.2，然后安装bert-as-service的server 和 client，在终端命令行通过输入
  ```terminal
   bert-serving-start -model_dir=tmp/uncased_L-12_H-768_A-12 -num_worker=2
  ```
- - 注：启动BERT服务，这里的经验教训是不能将num_worker设置成github中的4个并行，这样会导致这个内存溢出，使得整个mac崩溃，因此这里改为设置num_worker=2。
- 4. 通过输入终端的启动服务后，显示如下的界面表明启动服务成功。
![BERT-START](bert-start.png)
- 5. 打开另一个终端，打开python3解释器，我们输入以下python脚本
- 
  ```python
    from bert_serving.client import BertClient
    bc = BertClient()
  ```
  则在之前的老终端会启动一个client服务，如下图所示：
![BERT-CONFIG](BERT-Config.png)

- 继续在刚打开的新终端中输入python脚本
- 
   ```python
    bc.encode(['First do it'])
   ```
- 则返回一个bert生成关于英文语句"First do it"的768维句向量 
  如下图所示：
![WORD_EMBEDDING_1](word_embedding_1.png)

- 这只是一个句子，如果对于多个句子，通过``` ||| ```连接这些句子，从而得到多句的句向量表示，我们输入python脚本
- 
  ```python
  bc.encode(['First do it ||| then do it right ||| then do it better'])
  ```
  得到如下图所示的768维句向量
![WORD_EMBEDDING_2](word_embedding_2.png) 

- 6. 那么对于较为复杂的日文句子，也能尝试得到768维句向量，以日本较为流行的skyticket网站上的租房信息为例，输入以下的python脚本
- 
  ```python
    bc.encode(['全部で 84 ある冷房完備の部屋で、ご滞在をお楽しみください ||| 部屋ではWiFi (無料)をご利用いただけます ||| セーフティボックスをご利用いただけ、ハウスキーピング サービスは、リクエストにより行われます ||| 全部で 66 ある冷房完備の部屋には冷蔵庫、電子レンジなどが備わっており、ゆっくりおくつろぎいただけます ||| 有線インターネット アクセス / WiFi を無料でお使いいただけるほか、薄型テレビをご覧いただけます ||| シャワー付き浴槽のある専用バスルームには、深めの浴槽、バスアメニティ (無料)が備わっています ||| 東京 23 区の中央にあるホテルリブマックス東京新富町に泊まれば、東京国際フォーラム、キッザニア東京まで車で 5 分で行けます ||| このホテルは、歌舞伎座まで 0.9 km、皇居まで 2.1 km の場所にあります'])
  ```

  里面有日语汉字、片假名、数字、英文等，测试其生成的句向量能力，发现文本过长，虽然能够生成句向量，但是由于默认是最长长度是25个单词长度，因此这种长文本被从右截断为25个单词长度，生成的句向量准确性会有所降低，丢失了大量的语义信息，并且在运行过程中会报出警告信息```UserWarning: some of your sentences have more tokens than "max_seq_len=25" set on the server, as consequence you may get less-accurate or truncated embeddings.here is what you can do:
  disable the length-check by create a new "BertClient(check_length=False)" when you do not want to display this warning
  or, start a new server with a larger "max_seq_len"
  '- or, start a new server with a larger "max_seq_len"' % self.length_limit)```。
- 这种复杂的长文本句向量被截取25个单词之后，仍然为768维，其结果如下图所示，丢失了前面300多个单词的句法信息，准确性大大降低。
  
  ![WORD_EMBEDDING_3](word_embedding_3.png) 

- 7. 尝试将最长的句子设置为120，这样就超过了上面最长日语句子的长度，应该能够获得较为准确的句向量信息。在终端命令行通过输入
  ```terminal
   bert-serving-start -max_seq_len=120 -model_dir=tmp/uncased_L-12_H-768_A-12 -num_worker=2
  ```
  然后再次按照第6步的做法输入日文文档```全部で 84 ある冷房完備の部屋で、ご滞在をお楽しみください 。部屋ではWiFi (無料)をご利用いただけます 。 セーフティボックスをご利用いただけ、ハウスキーピング サービスは、リクエストにより行われます 。 全部で 66 ある冷房完備の部屋には冷蔵庫、電子レンジなどが備わっており、ゆっくりおくつろぎいただけます 。 有線インターネット アクセス / WiFi を無料でお使いいただけるほか、薄型テレビをご覧いただけます。シャワー付き浴槽のある専用バスルームには、深めの浴槽、バスアメニティ (無料)が備わっています 。東京 23 区の中央にあるホテルリブマックス東京新富町に泊まれば、東京国際フォーラム、キッザニア東京まで車で 5 分で行けます。 このホテルは、歌舞伎座まで 0.9 km、皇居まで 2.1 km の場所にあります```得到句向量如下图所示，没有提示警告信息。
  ![WORD_EMBEDDING_4](word_embedding_4.png) 

- 8. 为了使用返回简体中文的句向量，采用Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters的BERT-Base, Chinese版本作为预训练模型保存在tmp文件夹的tmp/chinese_L-12_H-768_A-12中，启动bert-service并调用这个预训练模型，考虑到中文语句不会像日文或者英文那么冗长，因此最长的句子长度设置为60个字已经较为足够。在终端的命令如下：
  ```terminal
   bert-serving-start -max_seq_len=60 -model_dir=tmp/chinese_L-12_H-768_A-12 -num_worker=2
  ```
- 尝试将马保国同志的经典语录```他说他是乱打的，他可不是乱打的啊，正蹬，鞭腿，左刺拳，训练有素，后来他说他练过三四年泰拳，啊，看来是有备而来！这两个年轻人不讲武德，来骗！来偷袭！我六十九岁的老同志，这好吗？这不好，我劝这位年轻人好自为之，好好反思，以后不要再犯这样的聪明，小聪明啊，武林要以和为贵，要讲武德，不要搞窝里斗，谢谢朋友们！```作为测试，
输入以下python脚本
  ```python
    bc.encode(['他说他是乱打的，他可不是乱打的啊，正蹬，鞭腿，左刺拳，训练有素 ||| 后来他说他练过三四年泰拳，啊，看来是有备而来 ||| 这两个年轻人不讲武德，来骗 ||| 来偷袭 ||| 我六十九岁的老同志，这好吗 ||| 这不好，我劝这位年轻人好自为之，好好反思，以后不要再犯这样的聪明 ||| 小聪明啊，武林要以和为贵，要讲武德，不要搞窝里斗，谢谢朋友们 ||| '])
  ```
  通过bert-as-service提取到相应的中文句向量如下图所示：
  ![WORD_EMBEDDING_5](word_embedding_5.png) 