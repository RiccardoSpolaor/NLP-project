- ~Rewrite code of model and training~
- ~Add seed for deterministic results~
- ~Add validation inside the training loop~


- ~plot delle lunghezze e taglio al 95% che ti accelera il training~
- ~modifica della definizione del modello, bastava chiamare from pretrained passandogli il numero delle classi~
- ~prendere un modello grosso (I modelli piccoli di solito funzionano peggio) e fare pochissimo allenamento (già alla seconda epoca va in overfit di solito) e fare validation ogni tot batch e early stopping (come se fosse zero shot quasi)~
- ~mettere due lr diversi per la parte prima del modello e per la testa di classificazione~
- ~modifica della funzione di loss, come minimo ripensando per lo sbilanciamento, ma proverei anche ad implementare la focal loss (basta aggiungere dentro un termine logaritmico)~