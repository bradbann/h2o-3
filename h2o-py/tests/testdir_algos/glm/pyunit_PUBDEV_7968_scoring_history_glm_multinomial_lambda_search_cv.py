from __future__ import division
from __future__ import print_function
from past.utils import old_div
import sys
sys.path.insert(1, "../../../")
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator as glm

# In this test, I will check and make sure the scoring history metrics of GLM with/without Lambda Search
# will contain the following: timestamp, duration, training_rmse, training_logloss, training_auc, training_pr_auc,
# training_classification_error.
def test_glm_scoring_history_multinomial():
    print("Preparing dataset....")
    h2o_data = h2o.import_file(
        pyunit_utils.locate("smalldata/glm_test/multinomial_10_classes_10_cols_10000_Rows_train.csv"))
    h2o_data["C1"] = h2o_data["C1"].asfactor()
    h2o_data["C2"] = h2o_data["C2"].asfactor()
    h2o_data["C3"] = h2o_data["C3"].asfactor()
    h2o_data["C4"] = h2o_data["C4"].asfactor()
    h2o_data["C5"] = h2o_data["C5"].asfactor()
    h2o_data["C11"] = h2o_data["C11"].asfactor()
    splits_frames = h2o_data.split_frame(ratios=[.8], seed=1234)
    train = splits_frames[0]
    valid = splits_frames[1]

    print("Building model with score_each_iteration turned on, with lambda search.")
    h2o_model_score_each = glm(family="multinomial", score_each_iteration=True, lambda_search=True, nlambdas=10)
    h2o_model_score_each.train(x=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], y="C11", training_frame=train,
                                  validation_frame=valid)
    print("Building model with score_interval=1.  Should generate same model as score_each_iteration turned on.")
    h2o_model = glm(family="multinomial", score_iteration_interval=1, lambda_search=True, nlambdas=10)
    h2o_model.train(x=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], y="C11", training_frame=train,
                               validation_frame=valid)

    print("Building model with score_each_iteration turned on, with lambda search and CV.")
    h2o_model_score_each_cv = glm(family="multinomial", score_each_iteration=True, lambda_search=True, nlambdas=10, 
                               nfolds=2, fold_assignment='modulo')
    h2o_model_score_each_each_cv.train(x=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], y="C11", training_frame=train,
                           validation_frame=valid)
    print("Building model with score_interval=1.  Should generate same model as score_each_iteration turned on, with "
          "lambda search and CV.")
    h2o_model_cv = glm(family="multinomial", score_iteration_interval=1, lambda_search=True, nlambdas=10, nfolds=2, 
                    fold_assignment='modulo')
    h2o_model_cv.train(x=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], y="C11", training_frame=train,
                validation_frame=valid)
    print("Done")

if __name__ == "__main__":
    pyunit_utils.standalone_test(test_glm_scoring_history_multinomial)
else:
    test_glm_scoring_history_multinomial()
