from lib.SecureBERT_functions import predict_mask, load_SecureBERT

tokenizer, model = load_SecureBERT()



while True:
    sent = input("Text here: \t")
    print("SecureBERT: ")
    predict_mask(sent, tokenizer, model)

    print("===========================\n")