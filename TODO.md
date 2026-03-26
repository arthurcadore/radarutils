7.3.3 Exercícios
Exercício: Faça a simulação de um radar monoestático (matlab ou phyton), considerando
o seguinte cenário: Alvo pontual estático + ruído AWGN.
• Pt = 100 kW - potência de transmissão
• fp = 3 GHz - frequência de transmissão
• Tp = 2µs - duração do pulso
• R = 60 km - distâncias do alvo ao radar.
• L = 6 dB - perdas do radar.
• Gt = Gr = 1.
• Alvo determinístico: σ = 1 m2
.
• Ruído AWGN no receptor: Assuma que o ruído no receptor é aditivo, gaussiano,
branco. Simule diferente valores de variância de ruído (Sugestão: N0 = 1, 10, 30, 70, 100).
Existem funções prontas para simular esse tipo de ruído (Dica: randn do matlab).
• Para cada valor de SNR (de N0), obtenha as curvas ROC (Pd × F AR).
• Considerem diferentes valores de threshold (suficientes para permitir a construção
de uma curva ROC). Em geral, entre 6 e 10 valores de threshold são suficientes para
se traçar uma curva ROC.
• Os valores de threshold podem ser obtidos empiricamente, baseado em testes.