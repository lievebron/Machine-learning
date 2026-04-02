# %% IMPORTS
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import  VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from worcliver.load_data import load_data
import shap
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.metrics import roc_curve, auc
from collections import Counter
from sklearn.model_selection import RepeatedStratifiedKFold
import matplotlib.pyplot as plt


# %% 1 CUSTOM TRANSFORMER VOOR CORR + SELECTK
# vooraf eerst definieren 
class CorrAndSelect(BaseEstimator, TransformerMixin): # definieert nieuw type object, BaseE voegt bouwstenen toe zodat je kunt gebruiken in functie zoals randomSearch (bv get_params), TransfMix voegt methode fit_transform toe
    def __init__(self, corr_threshold=0.9): # deze functie wordt 1 keer uitgevoerd op moment je klasse aanroept, lokale variable 0.9. Leeg object creeeren binnen het werkgeheugen
        self.corr_threshold = corr_threshold # koppelt getal aan dat object?
    
    def fit(self, X, y): #start leerfase van object self, x zijn de features en y zijn de labels
        if not isinstance(X, pd.DataFrame): #controle of binnenkomende data pd dataframe is (tabel met kolomnamen)
            X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]) # als het nog geen pd dataframe is dan dwing je het hier dr in en geeft die namen f0,f1,f2 enz

    ## Constanten verwijderen
        self.var_thresh_ = VarianceThreshold(threshold=0) # berekent variantie van elke colom, markeert de rijen met 0 verandering voor verwijdering
        X_const = pd.DataFrame(self.var_thresh_.fit_transform(X), # scant data en verwijdert direct constant kolommen, krijgt kale matrix zonder namen, pd om tabelstructuur te houden
                           columns=X.columns[self.var_thresh_.get_support()]) # get support geeft lijst met true of false voor elke kolom, zo selecteer je alleen namen van kolommen die NIET verwijderd zijn en plak je deze terug in nieuwe tabel X_const
        self.const_kept_features_ = X_const.columns # overgebleven kolommen/features opgeslagen in self

    # Correlatie filter
        corr_matrix = X_const.corr(method='spearman').abs() # correlatie berekenen tussen overgebleven features, adhv spearman en alles positief maken correlatie naar boven en naar bedenen even kut
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) # Deze regel "maskeert" de onderste helft en de diagonaal. Je houdt alleen de bovendriehoek van de tabel over. Zo voorkom je dat je features dubbel checkt of een feature met zichzelf vergelijkt.
        self.to_drop_ = [col for col in upper.columns if any(upper[col] > self.corr_threshold)] # als boven de threshold dan op de zwarte lijst, vormt dus de zwarte lijst
        X_filtered = X_const.drop(columns=self.to_drop_) # features op zwarte lijst worden verwijderd, kolom verwijderen welke? die in self.to_drop staan.

    # **Houd alle overgebleven features**
        self.features_ = X_filtered.columns # opslaan in geheugen
                                            # Wanneer je later de test-set verwerkt, weet de machine dankzij self.features_ precies welke x kolommen hij moet overhouden, zonder dat hij de hele berekening opnieuw hoeft te doen.

        return self # ketting verbreken, pipeline kan door met andere klasse
    
    def transform(self, X): # net waren we bij de leerfase nu bij de uitvoeringsfase
        if not isinstance(X, pd.DataFrame): # zorgen dat binnenkomen data nette tabel is
            X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]) # als geen nette tabel is naam geven f0,f1,f2 enz
        
        X_const = pd.DataFrame(self.var_thresh_.transform(X), # machine pakt de var_thresh_ uit zijn geheugen en verwijdert uit deze nieuwe data precies die kolommen die tijdens het trainen constant bleken te zijn
                               columns=X.columns[self.var_thresh_.get_support()]) # rekent dus niet opnieuw uit maar voert uit wat hier boven geleerd is
        X_filtered = X_const.drop(columns=self.to_drop_, errors='ignore') # features op de zwarte lijst er uitgooien, als een feature op zwarte lijst er niet in staat (wat error kan geven) gaat tie gewoon door
        X_selected = X_filtered[self.features_] # self.features waren hierboven overgebleven, filtert tabel zodat alleen die kolommen overblijven in de exacte volgorde van de training
        return X_selected # schone tabel klaar voor RF
    
    # %% 2 LOAD DATA
data = load_data()
X = data.select_dtypes(include=[np.number]) # alle kolommen met getallen overhouden, de rest weggooien, dus X tabel met alleen numeriek features
y = data['label'].map({'benign': 0, 'malignant': 1}) # feature genaamd label omzetten in cijfers, x en y zijn door pd aan elkaar verbonden

f2_scorer = make_scorer(fbeta_score, beta=2) # f2 score maken met beta=2


# %% 3 TRAIN/TEST SPLIT
X_trainval, X_test, y_trainval, y_test = train_test_split( # data opdelen in train en test set 80/20
    X, y, test_size=0.2, stratify=y, random_state=42 # stratify y geeft gelijke verhouding van ben/mal voor beide groepen en 42 zorgt dat het reproduceerbaar is doordat hierdoor altijd exacte dezelfde groepen maakt
)

# %% 4 NESTED CV
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # verdelen in 5 folds met gelijken verhoudingen, reproduceerbaar en shuffle aka kaarten schudden voor verdelen
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # hetzelde maar dan in 3

all_y_outer = [] # hierin wordt de def diagnose opgeslagen
all_y_outer_proba = [] # kansberekening van de diagnose 

feature_importances_list = [] # slaat per fold op welke features meest voorspellend zijn, van alle 5 folds
best_params_list = [] # optimale instelling van RF per fold, van alle 5 folds
fold_aucs = [] # AUC-score van validatie folds, alle 5

for outer_train_idx, outer_val_idx in outer_cv.split(X_trainval, y_trainval): # lijstje van indexen maken welke patient in welke groep mee doet
    X_outer_train = X_trainval.iloc[outer_train_idx].copy() # mri getallen pakken van 149 train patienten en kopie maken zodat er niks aangepast wordt in het bron bestand, eigen kladje om in te strepen
    X_outer_val = X_trainval.iloc[outer_val_idx].copy() # mri getallen pakken van 37 validatie patienten en kopie maken
    y_outer_train = y_trainval.iloc[outer_train_idx] # dezelfde split doen voor de diagnose zodat uitslagen bij de jusiste patient blijven, hier passen we nooit wat in aan dus geen kladje nodig
    y_outer_val = y_trainval.iloc[outer_val_idx]

    # Pipeline bepalen voor inner loop: correlatie + scaling + classifier
    pipeline = Pipeline([        
    ('feat_select', CorrAndSelect(corr_threshold=0.9)),
    ('scaler', RobustScaler()),
    ('clf', RandomForestClassifier(random_state=42))
])

    param_dist = {
    'clf__n_estimators': [100, 200, 300, 500], # officele namen met clf tag er aan zodat weet over welk deel in pipeline het gaat
    'clf__max_depth': [None, 5, 10, 20],
    'clf__min_samples_split': [2, 5, 10]
}

    # Inner loop GridSearchCV eerst definieren
    grid_search = RandomizedSearchCV(
        estimator=pipeline, # hele pipeline bekijken
        param_distributions=param_dist,  # menu met opties
        n_iter=30,           # 30 willekeurige combinaties worden geprobeerd
        cv=inner_cv,    # binnenste ring gebruiken 
        scoring=f2_scorer,  # rapportcijfer of basis van F2 hoe goed een combinatie van hyperparameters werkt
        n_jobs=1,    # standaard op -1 maar toen duurde het heel lang, gaat over computer processor
        random_state=42)
    
    grid_search.fit(X_outer_train, y_outer_train) # nu daadwerkelijk uitvoeren, je geeft de 4 folds dus hieruit moet tie er weer 3 maken

    best_params_list.append(grid_search.best_params_) # pakketje met de beste knoppen dmv append in de klaargezete bestparams zetten

    print(f"Outer fold best params: {grid_search.best_params_}") # printen van beste hyperparameters, voor iedere 5 folds

    # Beste model toepassen op outer validation fold
    best_model = grid_search.best_estimator_ # grid search 30 opties geprobeerd, de winnaar noemen we best model, winnaar wordt gekozen op basis van de F2 score gedefineerd hierboven
    y_outer_proba = best_model.predict_proba(X_outer_val)[:,1] # machine kijkt naar test fold, berekent voor ieder een probability, voor ROC-curve alleen kans op maligne meenemen (1)

    # Print aantal features na const + correlatie
    X_const = pd.DataFrame( # passen de zeeg die we eerder in het geheugen hebben gezet toe dus de contanten worden er uitgehaald
    best_model.named_steps['feat_select'].var_thresh_.transform(X_outer_train), # dankzij transform verwijder je niks uit een lijst maar maak je een nieuwe lijst dus hier niet error=ignore nodig
    columns=best_model.named_steps['feat_select'].const_kept_features_)

    X_corr = X_const.drop(columns=best_model.named_steps['feat_select'].to_drop_, errors="ignore") # hier geef je een lijst en vertel je welke verwijderen, als die er dan niet in staat gewoon door gaan
    X_selected = X_corr[best_model.named_steps['feat_select'].features_] # selecteren van juiste eindselectie en in juiste volgorde zetten!

    print(
    f"Outer fold: Features orig={X_outer_train.shape[1]}, " # N features in begin
    f"na const={X_const.shape[1]}, "    # N features na constant verwijderen
    f"na corr={X_corr.shape[1]}"    # N features na correlatie verwijderen
)

    all_y_outer.extend(y_outer_val) # extend want plakt het er direct als getallen achteraan, plakt na iedere ronde de getallen van de fold er achter aan van test set
    all_y_outer_proba.extend(y_outer_proba) # hetzelfde maar dan voor de probability score

    fold_auc = roc_auc_score(y_outer_val, y_outer_proba) # berekening van AUC, lijst met diagnoses van de testset en de bijhorende kansscore
    fold_aucs.append(fold_auc) # lege lijst boven in gedefinieerd, append is new item toegoeven achteraan aka AUC van de fold
    print(f"Outer fold AUC: {fold_auc:.3f}") # printen op 3 decimalen

    # Feature importances opslaan (alleen geselecteerde features)
    selected_features = best_model.named_steps['feat_select'].features_ # best_model is winnende pipeline uit grid search voor 1 fold, hier vraag je de namen van de features weer terug want die worden in RF niet meegenomen
    importances = best_model.named_steps['clf'].feature_importances_ # feature belangrijkheids score bij iedere feature ophalen
    feature_importances_list.append(pd.DataFrame({ # hier voeg je in de eerder gedefinieerde (lege) lijst voeg je in een nette tabel de feature naam en score toe
        'feature': selected_features,
        'importance': importances
    }))

# %% 5 EVALUATE NESTED CV
roc_auc = roc_auc_score(all_y_outer, all_y_outer_proba) # deze hebben we al gevuld met de AUC van alle 5 de folds nu een AUC score berekenen over alle 5 de folds de testfold samen
print(f"Nested CV ROC-AUC: {roc_auc:.3f}") # deze printen op 3 decimalen

from sklearn.metrics import fbeta_score

# probabilities -> class labels
y_pred = (np.array(all_y_outer_proba) >= 0.5).astype(int) # lijst maken met alle probabilities, als hoger dan 0.5 dan true/maligne, aka def diagnoses maken op basis van onze scores van het model

# F2 score
nested_f2 = fbeta_score(all_y_outer, y_pred, beta=2) # F2 score uitvoeren, gaat de gegeven lijst af next to de voorspelling en kijkt naar TP,TN,FP,FN
print(f"\nNested CV F2-score: {nested_f2:.3f}") # score printen op 3 decimalen


# %% 7 FINAL MODEL MET BESTE HYPERPARAMETERS VAN NESTED CV

# 1 Kies de beste hyperparameters van de outer folds
best_params = grid_search.best_params_ # pakt de winnende hyperparameters van de laatste fold (nr5), maakt niet uit uit welke fold hij deze namenlijst haalt, want in elke fold zijn dezelfde hyperparameters gebruikt om aan te draaien.

final_params = {} # nieuwe leeg kladje aan maken
for param in best_params_list[0].keys(): # voor de param in die lijst zoals hierboven aangehaald gaan we kijken/opnieuw instellen
    values = [p[param] for p in best_params_list] # lijst maken per parameter wat de beste waren voor de 5 folds
    final_params[param] = Counter(values).most_common(1)[0][0] # final parameters kiezen door te kijken welke het meeste voorkomt, lijstje maken welke op 1 staat en dan, [0] zegt eerste pakketje pakken uit lijst en daarna eerste getal pakken uit de haakjes

print("Consensus hyperparameters over alle outer folds:") # print de labels van de hyperparameters
print(final_params) # print de inhoud van de stemronde hierboven, dus van iedere hyperparameter de waarde van de beste

# 2 Maak final pipeline met dezelfde CorrAndSelect config en beste RF hyperparamaters toevoegen!
pipeline_final = Pipeline([
    ('feat_select', CorrAndSelect(corr_threshold=0.9)),
    ('scaler', RobustScaler()),
    ('clf', RandomForestClassifier(
        n_estimators=final_params['clf__n_estimators'],
        max_depth=final_params['clf__max_depth'],
        min_samples_split=final_params['clf__min_samples_split'],
        random_state=42
    ))
])

# 3 Fit pipeline op volledige trainval set 
pipeline_final.fit(X_trainval, y_trainval) # volledige 80% trainings set gebruiken, machine gaat leren adhv volledige trainingsdata

# 4 Transformeer test set en predict proba
X_test_transformed = pipeline_final.named_steps['feat_select'].transform(X_test) # de features verwijdert die hij hier boven heeft opgeslagen als slechte featers?
y_test_proba = pipeline_final.named_steps['clf'].predict_proba(X_test_transformed)[:,1] # model geeft waarschijnlijkheid op maligne, enkel deze kans bewaren

# 7 SHAP ANALYSE OP FINAL MODEL
X_model = pipeline_final.named_steps['feat_select'].transform(X_trainval)
X_model = pd.DataFrame(X_model, columns=pipeline_final.named_steps['feat_select'].features_)

explainer = shap.TreeExplainer(pipeline_final.named_steps['clf'])
shap_values = explainer.shap_values(X_model)

# SHAP array correct afhandelen
if isinstance(shap_values, list):
    shap_vals = np.array(shap_values[1])
elif shap_values.ndim == 3:
    shap_vals = shap_values[:, :, 1]
else:
    shap_vals = shap_values

mean_abs_shap = np.abs(shap_vals).mean(axis=0)
assert mean_abs_shap.shape[0] == X_model.shape[1], \
    f"Feature mismatch: {mean_abs_shap.shape[0]} vs {X_model.shape[1]}"

shap_importance = pd.DataFrame({
    "feature": X_model.columns,
    "mean_abs_shap": mean_abs_shap
}).sort_values(by="mean_abs_shap", ascending=False)

# Top 20 SHAP features plotten
plt.figure(figsize=(10,6))
plt.barh(shap_importance.head(20).feature[::-1], 
         shap_importance.head(20).mean_abs_shap[::-1])
plt.xlabel("Mean Absolute SHAP Value")
plt.title("Top 20 SHAP Feature Importance")
plt.show()

# 8 Print gekozen hyperparameters
print("\nGekozen hyperparameters RandomForest final model:")
for param, value in pipeline_final.named_steps['clf'].get_params().items():
    print(f"{param}: {value}")

# %%
# %% Nested ROC curve
from sklearn.metrics import roc_curve, auc

# ROC curve berekenen op outer fold predictions
fpr, tpr, thresholds = roc_curve(all_y_outer, all_y_outer_proba)
roc_auc_nested = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"Nested CV ROC (AUC = {roc_auc_nested:.3f})")
plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Nested Cross Validation Cor and Const selection")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

plt.show()


# %%
def Forest_const_cor(test=False):

    results = {
    "nested_auc_const_cor": float(roc_auc),
    "nested_f2_const_cor": float(nested_f2),
    "fold_aucs_const_cor": (fold_aucs),

}

    pipeline_final.fit(X_trainval, y_trainval)

    if test:
        test_scores = pipeline_final.predict_proba(X_test)[:,1]
        test_pred = (test_scores > 0.5).astype(int)
        test_auc = roc_auc_score(y_test, test_scores)
        test_f2 = fbeta_score(y_test, test_pred, beta=2)

        results["test_auc"] = float(test_auc)
        results["test_f2"] = float(test_f2)
        results["y_test"] = y_test 
        results["test_scores"] = test_scores  

    return results
