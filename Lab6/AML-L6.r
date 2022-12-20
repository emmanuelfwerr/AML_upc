####################################################################
# Advanced Machine Learning - MDS
# Lluís A. Belanche

# LAB 5: illustrating the SVM for text classification with a string kernel
#                      (one of the possible and simplest kernels for text)
#        illustrating the SVM with some basic diffusion kernels
# version of December 2022
####################################################################

# --- Environment setup ---
setwd('/Users/efwerr/GitHub/AML/Lab6')

set.seed(6046)
library(kernlab) # kernlab is a very powerful R library for kernel methods

#########################################
# Example 1
# Basic text kernel on Reuters news
#########################################

## We are going to use a slightly-processed version of the famous
## Reuters news articles dataset.  All articles with no Topic
## annotations are dropped. The text of each article is converted to
## lowercase, whitespace is normalized to single-spaces.  Only the
## first term from the Topic annotation list is retained (some
## articles have several topics assigned).  

## The resulting dataset is a list of pairs (Topic, News). We will use three topics for analysis: Crude Oil, Coffee and Grain-related news

## The resulting data frame contains 994 news items on crude oil,
## coffee and grain. The news text is the column "Content" and its
## category is the column "Topic". The goal is to create a classifier
## for the news articles.

## Note that we can directly read the compressed version (reuters.txt.gz). 
## There is no need to unpack the gz file; for local files R handles unpacking automagically

# as usual, you need to be in your local folder, or code something like this:
# setwd("~/UPC 2021-22/Docència/AA2 2021-22/Labos/NEW L5")

reuters <- read.table("reuters.txt.gz", header=T)

# We leave only three topics for analysis: Crude Oil, Coffee and Grain-related news
reuters <- reuters[reuters$Topic == "crude" | reuters$Topic == "grain" | reuters$Topic == "coffee",]

reuters$Content <- as.character(reuters$Content)    # R originally loads this as factor, so needs fixing
reuters$Topic <- factor(reuters$Topic)              # re-level the factor to have only three levels

levels(reuters$Topic)

length(reuters$Topic)

table(reuters$Topic)

## an example of a text about coffee
reuters[2,]

## an example of a text about grain
reuters[7,]

## an example of a text about crude oil
reuters[12,]

(N <- dim(reuters)[1])  # number of rows

# we shuffle the data first

reuters <- reuters[sample(1:N, N),]

# To deal with textual data we need to use a string kernel. Several such kernels are implemented in the "stringdot" method of the kernlab package. We shall use the simplest one: the n-spectrum kernel. The feature map represents the string as a multiset of its substrings of length n

# Example, for n=2 we have

# phi("ababc") = ("ab" -> 2, "ba" -> 1, "bc" --> 1, ... other -> 0)

# we can define a normalized 3-spectrum kernel (n is length)
k <- stringdot("spectrum", length=3, normalized=TRUE)

# Let's see some examples:

Frank.Sinatra <- "I did it my way"

k(Frank.Sinatra, Frank.Sinatra)

k(Frank.Sinatra, "He did it his way")

k(Frank.Sinatra, "She did it her way")

k(Frank.Sinatra, "Let's find our way out")

k(Frank.Sinatra, "Brexit means Brexit")

## We start by doing a kPCA

## first we define a modified plotting function 

plotting <-function (kernelfu, kerneln)
{
  xpercent <- eig(kernelfu)[1]/sum(eig(kernelfu))*100
  ypercent <- eig(kernelfu)[2]/sum(eig(kernelfu))*100
  
  plot(rotated(kernelfu), col=as.integer(reuters$Topic),
       main=paste(paste("Kernel PCA (", kerneln, ")", format(xpercent+ypercent,digits=3)), "%"),
       xlab=paste("1st PC -", format(xpercent,digits=3), "%"),
       ylab=paste("2nd PC -", format(ypercent,digits=3), "%"))
}

## Create a kernel matrix using 'k' as kernel (it takes a couple of minutes)

k <- stringdot("spectrum", length=5, normalized=TRUE)
K <- kernelMatrix(k, reuters$Content)
dim(K)

K[2,2]

K[2,3:10]

## Plot the result using the first 2 PCs (we can add colors for the two classes)

kpc.reuters <- kpca (K, features=2, kernel="matrix")
plotting (kpc.reuters,"5 - spectrum kernel")

## finally add a legend
legend("bottomleft", legend=c("crude oil", "coffee","grain"),    
       pch=c(1,1),                    # gives appropriate symbols
       col=c("red","black", "green"), # gives the correct color
       bg="transparent",              # makes the legend transparent
       cex = 0.7)                     # legend size

## We can also train a SVM using this kernel matrix in the training set

## First we should split the data into learning (2/3) and test (1/3) parts
ntrain <- round(N*2/3)     # number of training examples
tindex <- sample(N,ntrain) # indices of training examples
  
## The fit a SVM in the train part
svm1.train <- ksvm (K[tindex,tindex],reuters$Topic[tindex], type="C-svc", kernel='matrix')

## and make it predict the test part

## Let's call SV the set of obtained support vectors

## Then it becomes tricky. We must compute the test-vs-SV kernel matrix
## which we do in two phases:

# First the test-vs-train matrix
testK <- K[-tindex,tindex]
# then we extract the SV from the train
testK <- testK[,SVindex(svm1.train),drop=FALSE]

# Now we can predict the test data
# Warning: here we MUST convert the matrix testK to a 'kernelMatrix'
y1 <- predict(svm1.train,as.kernelMatrix(testK))

table (pred=y1, truth=reuters$Topic[-tindex])

cat('Error rate = ',100*sum(y1!=reuters$Topic[-tindex])/length(y1),'%')

## now we define a 3D plotting function

library("rgl")
open3d()

plotting3D <-function (kernelfu, kerneln)
{
  xpercent <- eig(kernelfu)[1]/sum(eig(kernelfu))*100
  ypercent <- eig(kernelfu)[2]/sum(eig(kernelfu))*100
  zpercent <- eig(kernelfu)[3]/sum(eig(kernelfu))*100
  
  # resize window
  par3d(windowRect = c(100, 100, 612, 612))
  
  plot3d(rotated(kernelfu), 
         col  = as.integer(reuters$Topic),
         xlab = paste("1st PC -", format(xpercent,digits=3), "%"),
         ylab = paste("2nd PC -", format(ypercent,digits=3), "%"),
         zlab = paste("3rd PC -", format(zpercent,digits=3), "%"),
         main = paste("Kernel PCA"), 
         sub = "red - crude oil | black - coffee | green - grain",
         top = TRUE, aspect = FALSE, expand = 1.03)
 }

kpc.reuters <- kpca (K, features=3, kernel="matrix")
plotting3D (kpc.reuters,"5 - spectrum kernel")

# It can be seen that the black dots ('coffee') are quite well separated from the rest; this was not
# apparent in the 2D plot and explains the very good predictive performance

#########################################
# Example 2
# Basic diffusion kernel on a toy graph
#########################################

library(igraph)
library(diffuStats) # if in trouble, go to https://www.bioconductor.org/packages/release/bioc/html/diffuStats.html
library(expm)

options (digits=3)

(g <- graph_from_literal (Fred-Bobby-Georgina-Fred, Fred-Charlie, Bobby-David-Anne-Georgina))

# Creation of a toy similarity measure
S <- diag(6)
names <- c("Fred","Bobby","Georgina","Charlie","David","Anne")
dimnames (S) <- list(names,names)

# Strength of relationships
S["Fred","Bobby"] <- 0.1
S["Fred","Georgina"] <- 0.3
S["Fred","Charlie"] <- 0.2
S["Bobby","Georgina"] <- 0.9
S["Bobby","David"] <- 0.4
S["Georgina","Anne"] <- 0.7
S["David","Anne"] <- 0.5

# make it symmetric

(S[lower.tri(S)] = t(S)[lower.tri(S)])

# add the node-node similarities to the graph
g <- graph_from_adjacency_matrix(S * 
                                   as.matrix(get.adjacency(g, type="both")), mode="undirected", weighted=TRUE)

# plot it
plot(g, edge.label=E(g)$weight)

eigen(S, only.values = TRUE)$values # S need not be psd

lambda <- 0.5
(K <- expm(lambda*S))

eigen(K, only.values = TRUE)$values # K is pd

# but better normalize it

normalize.kernel = function(K)
{
  k = 1/sqrt(diag(K))
  K * (k %*% t(k))
}

(K.n <- normalize.kernel(K))

# this step did not affect the relationships substantially
cor (rowSums(S), rowSums(K.n))


#########################################
# Example 3
# Community detection in realistic graphs
#########################################

# A community is a set of nodes with many edges inside the community and few edges between outside it (i.e. between the community itself and the rest of the graph.)

# The idea is first to find communities and then try to create a classifier to assign new webpages to the right community
# We work with a randomly generated graph:

g <- generate_graph(fun_gen = igraph::barabasi.game,
                    param_gen = list(n = 100, m = 3, directed = FALSE),
                    seed = 1)

# This function tries to find communities in graphs via a spin-glass model and simulated annealing

com <- cluster_spinglass(g, spins=3)
V(g)$color <- com$membership
g <- set_graph_attr(g, "layout", layout_with_kk(g))

plot(g, vertex.label.dist=1)
colorets <- c("skyblue2","orange3","green4")
legend("bottomleft",
       c('Statistics','CompSci','Math'),
       col=colorets,pch=1,text.col=colorets,
       bg = "transparent", cex = 0.7)

K.diff <- diffusionKernel(g, 1)
is_kernel(K.diff)
hist(eigen(K.diff, only.values = TRUE)$values, breaks = 30, main = "Eigenvalue histogram")

K.diff[1:7,1:7]

K.norm <- normalize.kernel(K.diff)

is_kernel(K.norm)

K.norm[1:7,1:7]

K.svm <- as.kernelMatrix(K.norm)
t <- factor(V(g)$color)

# save(K.norm,t, file="graph-kernels")
load(file="graph-kernels")

K.svm <- as.kernelMatrix(K.norm)

# LOOCV error
ksvm (K.svm, y=t, cross=length(t))
