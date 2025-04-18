from test_train_sets import *
from features import full_bert_features
from EmbeddingGetter import get_embeddings
from xgboost import XGBRegressor
import numpy as np
from evaluate import evaluate


def get_set(split, k=17):
    bert_embeddings = []
    sentiment_overall_embeddings = []
    sentiment_excitment_embeddings = []
    prosodic_overall_embeddings = []
    prosodic_excitment_embeddings = []
    overall_scores = []
    excitment_scores = []
    for i in split:
        sentiment_overall_embeddings.append(
            get_embeddings(i, "sentiment", "overall", k, ".").cpu().detach().numpy()
        )
        sentiment_excitment_embeddings.append(
            get_embeddings(i, "sentiment", "excited", k, ".").cpu().detach().numpy()
        )

        bert_embeddings.append(full_bert_features(i, True).cpu().detach().numpy())
        prosodic_overall_embeddings.append(
            get_embeddings(i, "prosodic", "overall", k, ".").cpu().detach().numpy()
        )
        prosodic_excitment_embeddings.append(
            get_embeddings(i, "prosodic", "excited", k, ".").cpu().detach().numpy()
        )

        overall_scores.append(get_score(i, "Overall"))
        excitment_scores.append(get_score(i, "Excited"))

    bs_overall = [
        np.concatenate([b, s])
        for b, s in zip(bert_embeddings, sentiment_overall_embeddings)
    ]

    bs_excitment = [
        np.concatenate([b, s])
        for b, s in zip(bert_embeddings, sentiment_excitment_embeddings)
    ]

    bsp_overall = [
        np.concatenate([b, s, p])
        for b, s, p in zip(
            bert_embeddings, sentiment_overall_embeddings, prosodic_overall_embeddings
        )
    ]

    bsp_excitment = [
        np.concatenate([b, s, p])
        for b, s, p in zip(
            bert_embeddings,
            sentiment_excitment_embeddings,
            prosodic_excitment_embeddings,
        )
    ]

    return (
        np.array(bert_embeddings),
        np.array(sentiment_overall_embeddings),
        np.array(sentiment_excitment_embeddings),
        np.array(prosodic_overall_embeddings),
        np.array(prosodic_excitment_embeddings),
        np.array(bs_overall),
        np.array(bs_excitment),
        np.array(bsp_overall),
        np.array(bsp_excitment),
        np.array(overall_scores),
        np.array(excitment_scores),
    )


def train_model(X_train, Y_train):
    model = XGBRegressor()

    model.fit(X_train, Y_train)

    return model


def evaluate_model(model, X_test, Y_test):
    y_pred = model.predict(X_test)

    evaluate(Y_test, y_pred)


if __name__ == "__main__":
    num_features = [5, 10, 15, 17]
    train_set = get_train_set(0) + get_train_set(1) + get_train_set(2)
    val_set = get_train_set(3)
    test_set = get_test_set()

    #################### Training #######################
    # All of the commented out code is for tunning k and training
    # (
    #     bert_embeddings_train,
    #     _,
    #     _,
    #     _,
    #     _,
    #     _,
    #     _,
    #     _,
    #     _,
    #     overall_scores_train,
    #     excitment_scores_train,
    # ) = get_set(train_set)
    # (
    #     bert_embeddings_val,
    #     _,
    #     _,
    #     _,
    #     _,
    #     _,
    #     _,
    #     _,
    #     _,
    #     overall_scores_val,
    #     excitment_scores_val,
    # ) = get_set(val_set)

    # print(f"Bert Overall")
    # model = train_model(bert_embeddings_train, overall_scores_train)
    # evaluate_model(model, bert_embeddings_val, overall_scores_val)

    # print(f"Bert Excited")
    # model = train_model(bert_embeddings_train, excitment_scores_train)
    # evaluate_model(model, bert_embeddings_val, excitment_scores_val)

    for k in num_features:
        pass
        # (
        #     bert_embeddings_train,
        #     sentiment_overall_embeddings_train,
        #     sentiment_excitment_embeddings_train,
        #     prosodic_overall_embeddings_train,
        #     prosodic_excitment_embeddings_train,
        #     bs_overall_train,
        #     bs_excitment_train,
        #     bsp_overall_train,
        #     bsp_excitment_train,
        #     overall_scores_train,
        #     excitment_scores_train,
        # ) = get_set(train_set, k)
        # (
        #     bert_embeddings_val,
        #     sentiment_overall_embeddings_val,
        #     sentiment_excitment_embeddings_val,
        #     prosodic_overall_embeddings_val,
        #     prosodic_excitment_embeddings_val,
        #     bs_overall_val,
        #     bs_excitment_val,
        #     bsp_overall_val,
        #     bsp_excitment_val,
        #     overall_scores_val,
        #     excitment_scores_val,
        # ) = get_set(val_set, k)

        # print(f"Sentiment Overall  {k}")
        # model = train_model(sentiment_overall_embeddings_train, overall_scores_train)
        # evaluate_model(model, sentiment_overall_embeddings_val, overall_scores_val)

        # print(f"Sentiment Excited  {k}")
        # model = train_model(
        #     sentiment_excitment_embeddings_train, excitment_scores_train
        # )
        # evaluate_model(model, sentiment_excitment_embeddings_val, excitment_scores_val)

        # print(f"Prosodic Overall  {k}")
        # model = train_model(prosodic_overall_embeddings_train, overall_scores_train)
        # evaluate_model(model, prosodic_overall_embeddings_val, overall_scores_val)

        # print(f"Prosodic Excited  {k}")
        # model = train_model(prosodic_excitment_embeddings_train, excitment_scores_train)
        # evaluate_model(model, prosodic_excitment_embeddings_val, excitment_scores_val)

        # print(f"BS Overall  {k}")
        # model = train_model(bs_overall_train, overall_scores_train)
        # evaluate_model(model, bs_overall_val, overall_scores_val)

        # print(f"BS Excited  {k}")
        # model = train_model(bs_excitment_train, excitment_scores_train)
        # evaluate_model(model, bs_excitment_val, excitment_scores_val)

        # print(f"BSP Overall  {k}")
        # model = train_model(bsp_overall_train, overall_scores_train)
        # evaluate_model(model, bsp_overall_val, overall_scores_val)

        # print(f"BSP Excited  {k}")
        # model = train_model(bsp_excitment_train, excitment_scores_train)
        # evaluate_model(model, bsp_excitment_val, excitment_scores_val)

    ##################Testing################################
    # Unimodal Test Overall
    (
        _,
        _,
        _,
        _,
        _,
        bs_overall_train,
        _,
        _,
        _,
        overall_scores_train,
        _,
    ) = get_set(train_set, 10)
    (
        _,
        _,
        _,
        _,
        _,
        bs_overall_test,
        _,
        _,
        _,
        overall_scores_test,
        _,
    ) = get_set(test_set, 10)

    model = train_model(bs_overall_train, overall_scores_train)
    evaluate_model(model, bs_overall_test, overall_scores_test)

    # Unimodal Test Excitment
    (
        _,
        _,
        sentiment_excitment_embeddings_train,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        excitment_scores_train,
    ) = get_set(train_set, 15)
    (
        _,
        _,
        sentiment_excitment_embeddings_test,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        excitment_scores_test,
    ) = get_set(test_set, 15)

    model = train_model(sentiment_excitment_embeddings_train, excitment_scores_train)
    evaluate_model(model, sentiment_excitment_embeddings_test, excitment_scores_test)

    # Multimodal Test Overall
    # Lucia, these are the models that you should do explainers on
    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        bsp_overall_train,
        _,
        overall_scores_train,
        _,
    ) = get_set(train_set, 10)

    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        bsp_overall_test,
        _,
        overall_scores_test,
        _,
    ) = get_set(test_set, 10)

    model = train_model(bsp_overall_train, overall_scores_train)
    evaluate_model(model, bsp_overall_test, overall_scores_test)

    # Lucia: Explainer code here

    # MultiModal Test Excitment
    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        bsp_excitment_train,
        _,
        excitment_scores_train,
    ) = get_set(train_set, 15)

    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        bsp_excitment_test,
        _,
        excitment_scores_test,
    ) = get_set(test_set, 15)

    model = train_model(bsp_excitment_train, excitment_scores_train)
    evaluate_model(model, bsp_excitment_test, excitment_scores_test)

    # Lucia: Explainer code goes here
