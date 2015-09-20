The code (demo) is about the paper "Tag-Weighted Dirichlet Allocation"
============================================================
The paper is at http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6729528 <br/>
Author: Shuangyin Li, Guan Huang, Ruiyang Tan, Rong Pan <br/>
Sun Yat-sen University <br/>
<br/>
Any question about code please contact us by emails.<br/>
shuangyinli AT cse.ust.hk<br/>
panr AT sysu.edu.cn.<br/>

License
------------------------------------------------------------
Copyright 2013 Shuangyin Li, Guan Huang, Ruiyang Tan, Rong Pan <br/>
Licensed under the Apache License, Version 2.0 (the "License"); <br/>
you may not use this file except in compliance with the License.  <br/>
You may obtain a copy of the License at <br/>
<br/>
    http://www.apache.org/licenses/LICENSE-2.0 <br/>
<br/>
Unless required by applicable law or agreed to in writing, software <br/>
distributed under the License is distributed on an "AS IS" BASIS, <br/>
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. <br/>
See the License for the specific language governing permissions and <br/>
limitations under the License. <br/>

Install
-------------------------------------------------------------
```
cd src/ && make
```


Usage
-------------------------------------------------------------
###Input file format: <br/>
DocNumLabels label1 label2 ... @ DocNumWords word1 word2 ...<br/>
DocNumLabels label1 label2 ... @ DocNumWords word1 word2 ...<br/>
DocNumLabels label1 label2 ... @ DocNumWords word1 word2 ...<br/>

Each row represent one document with labels. DocNumLables means the number labels of document. DocNumWords means the number words of document. Each label is integer and represent one label. Each word is integer and represent one word.<br/>
<br/>
demo/twda.demo.input is a simple demo input file.<br/>
demo/label.txt is the label dictionary file. The word in row 1 means the label0.<br/>
demo/words.dic is the word dictionary file.<br/>
<br/>
<br/>
###Training:<br/>
```
./twda est <input data file> <setting.txt> <num_topics> <model save dir>
```
Example: <br/>
```
./src/twda est demo/twda.demo.input src/setting.txt 10 demo/model
```
Some model training parameters are set in the file "setting.txt".

###Inference:<br/>
```
./twda inf <input data file> <setting.txt> <model dir> <prefix> <output dir>
```
Example:  <br/>
```
./src/twda inf demo/twda.demo.input src/setting.txt demo/model/ final demo/output/
```
We can get the doc-topics-dis.txt file in output dir. The file indicates the topic distribution in input data file. The values in the file should be exp(.) so that we can konw that exact probablility.
