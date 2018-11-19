<h3>Summary</h3>

<p>In this project, my goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants.
 In order to predict test data, I divided the original training data into 75% model training data and 25% model test data.<br/>
 I created a random forest model and a support vector machine model and the accuracy was random forest: 0.9978, support vector machine: 0.948, so we chose the random forest model and predicted test data. </p>

<h3>Background</h3>

<p>Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <a href="http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har">http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har</a> (see the section on the Weight Lifting Exercise Dataset).</p>

<h3>Setup</h3>

<pre><code class="r setup, include=TRUE">knitr::opts_chunk$set(echo = TRUE)
Sys.setlocale(&quot;LC_TIME&quot;,&quot;us&quot;)
set.seed(1234)
</code></pre>

<pre><code class="r , echo=FALSE">setwd(&quot;C:/Users/ito/Desktop/cousera/datasience/8.Practical Machine Learning/pa&quot;)
</code></pre>

<h3>Version information about R</h3>

<pre><code class="r lib, message=FALSE, warning=FALSE">library(lattice)
library(ggplot2)
library(caret)
library(gbm)
library(elasticnet)
library(e1071)
library(randomForest)
sessionInfo()
</code></pre>

<h3>Getting Data</h3>

<ul>
<li>The training data for this project are available <a href="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv">here</a></li>
<li>The test data are available <a href="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv">here</a><br/>
&ldquo;{r gettingdata,cache=TRUE}
training &lt;- read.csv(&quot;pml-training.csv&rdquo;)
testing &lt;- read.csv(&ldquo;pml-testing.csv&rdquo;)</li>
</ul>

<h1>Dimensions</h1>

<p>rbind(training = dim(training),testing = dim(testing))</p>

<pre><code>The data for this project come from [this source](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har). If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

### Processing Data
 First, remove The Near Zero variance variables . 
```{r processingdata,cache=TRUE}

#clean The Near Zero variance variables
nzv &lt;- nearZeroVar(training)
training_nzv &lt;- training[,-nzv]

# Dimensions
dim(training_nzv)
</code></pre>

<p>Next, remove the variables which almost is &ldquo;NA&rdquo;</p>

<pre><code class="r, cache=TRUE">na_mean &lt;- sapply(training_nzv, function(x) mean(is.na(x)))
na &lt;- (na_mean &gt; 0.95)

training_na &lt;- training_nzv[,!na]
# Dimensions &amp; names
dim(training_na);head(names(training_na),n=10)
</code></pre>

<p>For the first 5 rows we do not need.</p>

<pre><code class="r, cache=TRUE"># remove the columns 
# from &quot;X&quot; to &quot;cvtd_timestamp&quot; 
processing &lt;- training_na[-(1:5)]
dim(processing)
</code></pre>

<h3>the correlation heatmap</h3>

<pre><code class="r, cache=TRUE">matrix_cor &lt;- cor(processing[,-54])
heatmap(matrix_cor,Colv = NA,Rowv = NA,col = cm.colors(256))
</code></pre>

<p>Looking at the correlation matrix, we can find combinations of variables with some correlation. However, it seems that their influence is small.</p>

<h3>fitting a model</h3>

<p>Create a prediction model from the original training data.<br/>
First, 75% of the original training data is used as the training data of the model, and 25% is set as the test data.</p>

<pre><code class="r, cache=TRUE">inTrain = createDataPartition(processing$classe, p = 0.75,list=F)
trainingdata = processing[ inTrain,]
testingdata = processing[-inTrain,]

rbind(train=dim(trainingdata), test=dim(testingdata))
</code></pre>

<h3>Random Forest</h3>

<pre><code class="r, cache=TRUE">mod_rf &lt;- randomForest(classe~.,data=trainingdata, method = &quot;class&quot;)

pre_rf &lt;- predict(mod_rf, newdata = testingdata)

confusionMatrix(pre_rf, testingdata$classe)
</code></pre>

<p>As a result of testing with a random forest, the accuracy is 0.9978.</p>

<h3>Support Vector Machines</h3>

<pre><code class="r, cache=TRUE">
mod_svm &lt;- e1071::svm(classe~.,data=trainingdata)

pre_svm &lt;- predict(mod_svm, newdata = testingdata)

confusionMatrix(pre_svm, testingdata$classe)
</code></pre>

<p>As a result of testing with the support vector machine, the accuracy is 0.948, which is lower than the result of the random forest.</p>

<h3>Prediction of test data</h3>

<p>Predict test data using random forest that is more accurate than support vector machine.</p>

<pre><code class="r, cache=TRUE">pre_rf2 &lt;- predict(mod_rf, newdata = testing)
pre_rf2
</code></pre>
