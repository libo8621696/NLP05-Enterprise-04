from bert_serving.client import BertClient
bc = BertClient()

bc.encode(['First do it ||| then do it right ||| then do it better'])

bc.encode(['全部で 84 ある冷房完備の部屋で、ご滞在をお楽しみください ||| 部屋ではWiFi (無料)をご利用いただけます ||| セーフティボックスをご利用いただけ、ハウスキーピング サービスは、リクエストにより行われます'])