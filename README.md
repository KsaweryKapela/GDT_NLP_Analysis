# GDT_NLP_Analysis
Using BERT embeddings on open-ended text answers regarding gaming habits to predict Gaming Disorder Test scores.

BERT embeddings are generated in 'feature_extraction' folder, 
while predictive models fine-tuning can be found in 'shallow_models' and 'deep_models' sections.

'Helpers' contain helping functions used in whole repo and 'visualizations and basic statics'
stores all nessecarry additional analysis, such as word counting and basic statics.

As per study pre-registration, final employed model was Ridge Regression, with best results of
r = 0.476, which is a decent result taking into account pioneer work done and not the best data quality.

Model clearly lacks samples of higher GDT scorers, which can be seen in figure below.