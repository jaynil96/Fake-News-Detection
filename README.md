# Fake-News-Detection
<h2>About</h2>
<ul>
<li>This is an attempt to build a model to predict whether the passed news data is fake or legitamate </li>
<li>Passive Agressive Classifier was used to find the model, after preprocing the data using TfidfVectorizer</li>
<li>For a more comprehensive look at this model, have a look<a href = "https://doingdddm.blogspot.com/2020/06/fake-new-detection-model.html"> here </a></li>
</ul>
<h2>Features</h2>
<ul>
<li>Using PassiveAgressiveIdentifier to build a model
	<li>Using the Confusion Matrix, F1Score and Jaccard Score libraries to examin the created models</li>
	<li>Using Math.plotlib to plot the confusion matrix</li>
	<li>Examining the SVM and Data Pre-processing Techniques(TfidfVectorizer)</li>
</ul>
<h2>Future Additions</h2>
<ul>
<li>As stated in the blog, This is part 1 of a larger project, it would be interesting to have it take in RSS data</li>
	<li>The F1 score might be too high and may be an indication of an overfit for the data set. Examining and re-implementation is in the works</li>
</ul>
