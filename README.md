algunas cosas a tener en cuenta:
lo arme pensando que los datasets (tanto train val y test) se pasan por las dos funciones de data_cleanse y la de hmv
puede tener sentido evaluar sobre un dataset al que no se le eliminan outliers? quizas, no se
o sea, el flujo es:
raw --> data cleanse --> handle missing values --> pasado a numerico (df_to_numeric)

habria que tener en cuenta que hay variables que en un principio tenian un monton de vlaores Nan (como camara de retroceso, HP o traccion). Muchos valores inferidos, por lo que puede traer problemas.


