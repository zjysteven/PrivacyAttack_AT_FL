# Exploiting Adversarial Training models to compromise privacy



Adversarial Training (AT) is crucial for obtaining deep neural networks that are robust to adversarial attacks, yet
recent works found that it could also make models more vulnerable to privacy attacks. In this work, we further reveal
this unsettling property of AT by designing a novel privacy attack that is practically applicable to the privacy-sensitive
Federated Learning (FL) systems. Using our method, the attacker can exploit AT models in the FL system to accurately reconstruct users’ private training images even when the training batch size is large, despite that previously large batch training was thought to be able to protect the privacy.
