# References for the comments in `svm_training.py`

Each claim made in the comments/docstrings of [svm_training.py](svm_training.py) is listed with the
source that supports it. Line numbers refer to the current version of the file.

Status legend:
- ✅ **Supported**: the source states it directly.
- ⚠️ **Needs nuance**: true in spirit, but the comment overstates it or leaves out a condition.
- ❌ **Inaccurate / unsupported**: the comment is wrong, or there is no source for it as written.

---

## 1. Infrastructure (lines 6–19)

| Line | Claim | Status | Source |
|---|---|---|---|
| 6–11 | `matplotlib.use('Agg')` is a non-interactive backend that only writes files | ✅ | Matplotlib, *Backends*: https://matplotlib.org/stable/users/explanations/backends.html |
| 7–8 | A GUI backend can crash at exit with *"ICE default IO error handler doing an exit()"* | ✅ | libICE manual, ch. 13 *Error Handling*: the default IO error handler "print[s] an explanatory message to stderr and if the severity is fatal, call[s] exit with a nonzero value". https://xorg.freedesktop.org/archive/current/doc/libICE/ICElib.html |
| 8 | `errno = 32` | ✅ | errno 32 is `EPIPE` (broken pipe) on Linux: `man 3 errno`, https://man7.org/linux/man-pages/man3/errno.3.html |
| 9–10 | It caused failures on the "Lecun" server | — | Internal observation; no external source needed. (Typo: "running]".) |
| 16–19 | Paths resolved from `__file__`, not the cwd | ✅ | Python docs: `__file__` https://docs.python.org/3/reference/datamodel.html#module.__file__ and `os.path.abspath` https://docs.python.org/3/library/os.path.html#os.path.abspath |

## 2. Linear SVM model and objective (lines 22–56)

| Line | Claim | Status | Source |
|---|---|---|---|
| 22–23 | Hinge loss = max(0, 1 − y(wᵀx + b)), y ∈ {−1, 1} | ✅ | Cortes & Vapnik (1995) [1]; Hastie et al., *ESL*, §12.3.2, eq. 12.25 [2]; Wikipedia *SVM* (soft margin): https://en.wikipedia.org/wiki/Support_vector_machine#Soft-margin |
| 24 | Primal objective: 0.5‖w‖² + C · **mean**(hinge) | ⚠️ | The standard soft-margin primal uses the **sum** of slacks, C·Σξᵢ [1][3][7]. The **mean** form is a valid rescaling, so this code's `C` equals sklearn's `C` × n_samples. A C value from sklearn's grid therefore does **not** mean the same here. |
| 27–31 | A linear SVM is just w, b and the score w·x + b; what makes it an SVM is the hinge-loss training | ✅ | *ESL* §12.3.2 and Table 12.1 [2] write the SVM as "loss + penalty" with the hinge loss, next to other losses (logistic, squared error) on the same linear model. Shalev-Shwartz & Ben-David, *Understanding Machine Learning*, ch. 15 [4]. |
| 45–47 | The score is a "signed distance-like" value | ✅ | The signed distance to the hyperplane is (w·x + b)/‖w‖, *ESL* §4.5 [2]. The raw score is proportional to it. |

## 3. Standardization (lines 59–99)

| Line | Claim | Status | Source |
|---|---|---|---|
| 61–63 | Replacement for `StandardScaler().fit()` (z-score) | ✅ | sklearn `StandardScaler`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html |
| 64–66 | Standardizing matters for an SVM: features on larger scales dominate | ✅ | Hsu, Chang & Lin [5], §2.2: "The main advantage of scaling is to avoid attributes in greater numeric ranges dominating those in smaller numeric ranges." sklearn *SVM: Tips on Practical Use*: "Support Vector Machine algorithms are not scale invariant, so it is highly recommended to scale your data." https://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use. For penalized linear models generally, see *ESL* §3.4.1 [2]. |
| 69–71 | Fit on TRAINING data only, otherwise test information leaks | ✅ | sklearn *Common pitfalls*, §"Data leakage during pre-processing" (the warning covers `StandardScaler`): https://scikit-learn.org/stable/common_pitfalls.html. Kaufman et al. (2012) [12]. |
| 75–76 | A zero-variance feature gets std = 1 | ✅ | Same behaviour as sklearn: "If a variance is zero, we can't achieve unit variance, and the data is left as-is, giving a scaling factor of 1." (`StandardScaler`, *Attributes* → `scale_`) |
| 79 | `unbiased=False` | ✅ | Matches sklearn: "We use a biased estimator for the standard deviation, equivalent to `numpy.std(x, ddof=0)`." (`StandardScaler`, *Notes*) |
| 86–89 | Apply the SAME mean/std to train and test | ✅ | Hsu, Chang & Lin [5], §2.2: "we have to use the same method to scale both training and testing data." |

## 4. Stratified cross-validation (lines 102–143)

| Line | Claim | Status | Source |
|---|---|---|---|
| 104–107 | Stratified k-fold keeps class proportions per fold (like `StratifiedKFold`) | ✅ | sklearn `StratifiedKFold`: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html. Kohavi (1995) [9] found stratified CV generally better (lower bias and variance) than plain CV. |
| 107–110 | Without stratification, a fold can end up with no seizures, so sensitivity is undefined | ✅ | Forman & Scholz (2010) [10], on per-fold metrics becoming undefined or biased with rare positives. |
| 112–116 | Shuffles each class separately | ⚠️ | Correct as a description. But shuffling breaks the time order: neighbouring 1-s windows (strongly correlated in EEG) land in both training and validation folds, which makes the CV score optimistic. See Bergmeir & Benítez (2012) [13] and Roberts et al. (2017) [14] on CV with temporally dependent data. |

## 5. Training (lines 146–223)

| Line | Claim | Status | Source |
|---|---|---|---|
| 148–151 | Gradient descent on the soft-margin primal objective | ✅ | Solving the primal directly is well established: Chapelle (2007) [6]; Pegasos, Shalev-Shwartz et al. (2011) [8], uses stochastic sub-gradient descent. |
| 152–153 | "This is the same objective sklearn's `SVC(kernel='linear')` solves via quadratic programming" | ⚠️ | (a) Same objective **only up to the C rescaling** in §2 (sklearn's primal uses C·Σξᵢ, https://scikit-learn.org/stable/modules/svm.html#svc). (b) sklearn/libsvm solves the **dual** with an SMO-type decomposition method (Platt 1998 [15]; Chang & Lin 2011 [7]), not a generic QP solver. |
| 153–156 | Solved with Adam instead | ⚠️ | Adam: Kingma & Ba (2015) [11]. Note that the objective is **non-smooth** (the hinge has a kink), and Adam with a fixed 300 epochs and a constant learning rate has no guarantee of reaching the exact SVM optimum. Pegasos [8] gets its guarantees from a decreasing step size. The result is an *approximate* SVM. |
| 156 | "unlike a black-box QP solver" | ❌ | Opinion. libsvm's solver and all its options are fully documented [7]. Suggest removing it or rewording to "more flexible". |
| 159–161 | 0.5‖w‖² is the regularizer and pushes towards a wider margin | ✅ | The geometric margin is 2/‖w‖, so minimizing ‖w‖ maximizes it: Cortes & Vapnik [1]; *ESL* §12.2 [2]. |
| 162–165 | The hinge is 0 when y·score ≥ 1 and grows linearly otherwise, so only "hard" points contribute gradient | ⚠️ | True **for the hinge term only**: its sub-gradient is zero when the margin is above 1. These points are the support vectors [1][2]. The regularizer's gradient (w) still applies on every step regardless of the points. |
| 166–169 | Large C: fit harder, risk of overfitting; small C: wider margin, risk of underfitting | ✅ | *ESL* §12.2.1 [2]; sklearn *SVM* user guide, section on the C parameter: https://scikit-learn.org/stable/modules/svm.html. Hsu, Chang & Lin [5], §3.2 and Fig. 1. |
| 175–176 | "`C`: regularization strength" | ❌ | Backwards. C is the **inverse** of the regularization strength (sklearn `SVC`: "The strength of the regularization is inversely proportional to C."): https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html. The same wording appears at lines 294 and 325. Line 166 has it right. |
| 180–183 | Balanced weighting mirrors sklearn `class_weight='balanced'` | ✅ | sklearn formula `n_samples / (n_classes * np.bincount(y))`: https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html. The code's `total / (2 * n_c)` is that formula with n_classes = 2. In sklearn the weight scales C per class (*SVM → Unbalanced problems*), which is equivalent to weighting each hinge term. |
| 183–185 | Without weighting, the model tends to always predict the majority class | ✅ | He & Garcia (2009) [16]. For SVMs specifically, Veropoulos, Campbell & Cristianini (1999) [17] proposed different C values per class to control sensitivity and specificity. *Note:* in this pipeline the imbalance is limited, because `poincare_features.py` samples only 15 background windows per file. |

## 6. Prediction and metrics (lines 226–285)

| Line | Claim | Status | Source |
|---|---|---|---|
| 229–231 | Threshold the score at 0 | ✅ | The SVM decision function is sign(w·x + b) [1][2]. |
| 231–232 | `torch.no_grad()` because inference needs no gradients | ✅ | PyTorch `torch.no_grad`: https://pytorch.org/docs/stable/generated/torch.no_grad.html |
| 254–262 | Definitions of accuracy, sensitivity (positive-class recall) and specificity (negative-class recall) | ✅ | Altman & Bland (1994) [18]. sklearn `recall_score`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html |
| 257–259 | "Sensitivity is usually the metric that matters most … missing a seizure is worse than a false alarm" | ⚠️ / ❌ as written | Seizure-detection literature treats **both** as essential. Baumgartner, Koren & Rothmayer (2018) [19]: "low FARs are essential for the acceptance of an algorithm in a clinical setting, especially if the algorithm is applied as an alarm device", and specificity (false alarm rate) "still needs to be improved". The validation standard of Beniczky & Ryvlin (2018) [20] requires reporting sensitivity **and** false alarm rate. Shoeb & Guttag (2010) [21] report false detections per 24 h alongside sensitivity. Suggest rewording to "sensitivity is usually prioritized, but the false alarm rate decides clinical acceptance". |
| 271–273 | Sensitivity is undefined when there are no positives, specificity when there are no negatives | ✅ | Follows from the definitions (0/0) [18]. |

## 7. Patient-specific pipeline (lines 288–339)

| Line | Claim | Status | Source |
|---|---|---|---|
| 291–293 | One model per patient | ✅ | Patient-specific detectors on scalp EEG: Shoeb et al. (2004) [22]; Shoeb & Guttag (2010) [21], the CHB-MIT reference paper. Dataset: CHB-MIT on PhysioNet, https://physionet.org/content/chbmit/1.0.0/; Goldberger et al. (2000) [23]. |
| 298–301 | Chronological split, so the "test set is always in the future" | ⚠️ | Holding out future data is the right principle for dependent data [13]. **But in this code it is only approximate.** Windows are ordered by file, and within each file `poincare_features.py` stores all seizure windows first, then 15 background windows taken at random times. The windows of one seizure can also fall on both sides of the split. Shoeb & Guttag [21] instead used leave-one-record-out evaluation. |
| 302–303 | Fit the scaler on the training split only | ✅ | See §3 (sklearn pitfalls; [12]). |
| 304–308 | Pick C by grid search with k-fold CV on the training split | ✅ | Hsu, Chang & Lin [5], §3.2, including log-spaced C values ("exponentially growing sequences"). *ESL* §7.10 [2]. |
| 306–307, 311 | Score each fold by **sensitivity only** | ⚠️ | A design choice, not an established practice. With recall as the only criterion, false positives play no part in choosing C, and ties go to the smallest C. For imbalanced problems, He & Garcia [16] recommend measures that combine both classes, such as the G-mean (√(sens × spec)) or balanced accuracy. |
| 308–310 | (train score, validation score) per C gives the validation curve, which shows over/underfitting | ✅ | sklearn *Validation curves*: https://scikit-learn.org/stable/modules/learning_curve.html#validation-curve |
| 309–310 | "see module docstring guidance on reading over/underfitting" | ❌ | **The file has no module docstring**, so this reference points nowhere. Add the guidance or remove the reference. |
| 311–314 | Refit on the FULL training split with the best C; the CV models are discarded | ✅ | Hsu, Chang & Lin [5], §1.2: "Use the best parameter C and γ to train the whole training set". sklearn `GridSearchCV(refit=True)`: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html |
| 315–316 | Evaluate on the untouched test split | ✅ | Cawley & Talbot (2010) [24]: evaluating on data used for model selection biases the estimate. |

## 8. Other inline comments

| Line | Claim | Status | Source |
|---|---|---|---|
| 365 | `map_location` because the tensors are saved on the GPU | ✅ | PyTorch `torch.load`, `map_location`: https://pytorch.org/docs/stable/generated/torch.load.html |
| 489 | boxplot's `labels` was renamed `tick_labels` in Matplotlib 3.9 | ✅ | Matplotlib 3.9.0 API changes, *Deprecations*: "The parameter labels has been renamed to tick_labels". https://matplotlib.org/stable/api/prev_api_changes/api_changes_3.9.0.html |

---

## Bibliography

1. C. Cortes, V. Vapnik. **Support-vector networks.** *Machine Learning* 20, 273–297 (1995). https://doi.org/10.1007/BF00994018
2. T. Hastie, R. Tibshirani, J. Friedman. ***The Elements of Statistical Learning***, 2nd ed., Springer (2009). §3.4.1, §4.5, §7.10, §12.2–12.3. Free PDF: https://hastie.su.domains/ElemStatLearn/
3. scikit-learn User Guide, **Support Vector Machines** (SVC mathematical formulation; Tips on practical use; Unbalanced problems). https://scikit-learn.org/stable/modules/svm.html
4. S. Shalev-Shwartz, S. Ben-David. ***Understanding Machine Learning: From Theory to Algorithms***, Cambridge University Press (2014), ch. 15. Free PDF: https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/
5. C.-W. Hsu, C.-C. Chang, C.-J. Lin. **A Practical Guide to Support Vector Classification** (2003, updated 2025). https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
6. O. Chapelle. **Training a support vector machine in the primal.** *Neural Computation* 19(5), 1155–1178 (2007). https://doi.org/10.1162/neco.2007.19.5.1155
7. C.-C. Chang, C.-J. Lin. **LIBSVM: A library for support vector machines.** *ACM TIST* 2(3), 27 (2011). https://doi.org/10.1145/1961189.1961199
8. S. Shalev-Shwartz, Y. Singer, N. Srebro, A. Cotter. **Pegasos: primal estimated sub-gradient solver for SVM.** *Mathematical Programming* 127, 3–30 (2011). https://doi.org/10.1007/s10107-010-0420-4
9. R. Kohavi. **A study of cross-validation and bootstrap for accuracy estimation and model selection.** *IJCAI* (1995), 1137–1143. https://dl.acm.org/doi/10.5555/1643031.1643047
10. G. Forman, M. Scholz. **Apples-to-apples in cross-validation studies: pitfalls in classifier performance measurement.** *ACM SIGKDD Explorations* 12(1), 49–57 (2010). https://doi.org/10.1145/1882471.1882479
11. D. P. Kingma, J. Ba. **Adam: A method for stochastic optimization.** *ICLR* (2015). https://arxiv.org/abs/1412.6980
12. S. Kaufman, S. Rosset, C. Perlich, O. Stitelman. **Leakage in data mining: formulation, detection, and avoidance.** *ACM TKDD* 6(4), 15 (2012). https://doi.org/10.1145/2382577.2382579
13. C. Bergmeir, J. M. Benítez. **On the use of cross-validation for time series predictor evaluation.** *Information Sciences* 191, 192–213 (2012). https://doi.org/10.1016/j.ins.2011.12.013
14. D. R. Roberts et al. **Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure.** *Ecography* 40(8), 913–929 (2017). https://doi.org/10.1111/ecog.02881
15. J. Platt. **Sequential Minimal Optimization: A fast algorithm for training support vector machines.** Microsoft Research Tech. Report MSR-TR-98-14 (1998). https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/
16. H. He, E. A. Garcia. **Learning from imbalanced data.** *IEEE Trans. Knowledge and Data Engineering* 21(9), 1263–1284 (2009). https://doi.org/10.1109/TKDE.2008.239
17. K. Veropoulos, C. Campbell, N. Cristianini. **Controlling the sensitivity of support vector machines.** *Proc. IJCAI-99*, 281–288 (1999). PDF: https://sci2s.ugr.es/keel/pdf/algorithm/congreso/1999%20-%20Veropoulos%20-%20Controlling%20the%20Sensitivity%20of%20Support%20Vector%20Machines.pdf
18. D. G. Altman, J. M. Bland. **Diagnostic tests 1: sensitivity and specificity.** *BMJ* 308, 1552 (1994). https://doi.org/10.1136/bmj.308.6943.1552
19. C. Baumgartner, J. P. Koren, M. Rothmayer. **Automatic computer-based detection of epileptic seizures.** *Frontiers in Neurology* 9, 639 (2018). https://doi.org/10.3389/fneur.2018.00639
20. S. Beniczky, P. Ryvlin. **Standards for testing and clinical validation of seizure detection devices.** *Epilepsia* 59(S1), 9–13 (2018). https://doi.org/10.1111/epi.14049
21. A. Shoeb, J. Guttag. **Application of machine learning to epileptic seizure detection.** *Proc. ICML* (2010). PDF: https://archive.physionet.org/pn6/chbmit/shoeb-icml-2010.pdf
22. A. Shoeb et al. **Patient-specific seizure onset detection.** *Epilepsy & Behavior* 5(4), 483–498 (2004). https://doi.org/10.1016/j.yebeh.2004.05.005
23. A. L. Goldberger et al. **PhysioBank, PhysioToolkit, and PhysioNet.** *Circulation* 101(23), e215–e220 (2000). https://doi.org/10.1161/01.CIR.101.23.e215. Dataset: https://physionet.org/content/chbmit/1.0.0/
24. G. C. Cawley, N. L. C. Talbot. **On over-fitting in model selection and subsequent selection bias in performance evaluation.** *JMLR* 11, 2079–2107 (2010). https://jmlr.org/papers/v11/cawley10a.html
