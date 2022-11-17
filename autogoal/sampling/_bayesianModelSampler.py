
import builtins
from secrets import choice
from select import select
from autogoal.sampling import Sampler, best_indices
import random

class BayesianModelSampler(Sampler):
    def __init__(
        self,
        model = {},
        alpha : float = 0,
        exploration : bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._model = model
        self._alpha = alpha
        self._exploration = exploration
        self._updates = {}

    def _clamp(self, x, a, b):
        if x < a:
            return a
        if x > b:
            return b
        return x

    def _get_model_params(self, handle, default):
        if handle in self._model:
            return self._model[handle]
        else:
            self._model[handle] = default
            return default

    def _register_update(self, handle, result):
        self._updates[handle] = result



    # model: {
    # option: number_of_choosen
    # }
    # La posición de cada value corresponde con la posición del
    # número de veces que fue escogido
    def choice(self, options, handle=None):
        if handle is None:
            return super().choice(options, handle)

        # Obtenemos el diccionario asociado a esta distribución
        choice_dict = self._get_model_params(handle, {})

        # Creamos una lista en la que guardaremos todas las veces que hemos
        # devuelto cada opción
        number_of_choosen = [0] * len(options)

        # Por cada opción, vemos si se encuentra en el modelo,
        # si no se encuentra la agregamos
        for i in range(len(options)):
            if options[i] in choice_dict:
                number_of_choosen[i] = choice_dict[options[i]]
            else:
                choice_dict[options[i]] = 0
        
        # Si solamente hay una opción, devolvemos directamente esta
        if len(options) == 1:
            indice = 0
        else:
            # Si hay más de una, en caso de estar explorando, seleccionamos una opción
            # entre aquellas que menos veces hemos escogido, en caso contrario, escogemos
            # una entre aquellas que más veces. El rango de selección está dado por alfa
            if self._exploration:
                indices = best_indices(number_of_choosen, random.randint(1, max(1, int(len(number_of_choosen)/2))), False)        
            else:
                indices = best_indices(number_of_choosen, random.randint(1, max(1, int(len(number_of_choosen)/2))), True)
            selections_range = int(len(indices) * (1-self._alpha))

            indice = indices[random.randint(0, max(selections_range - 1, 0))]
        
        # Obtenemos la opción escogida
        selected_value = options[indice]

        # Actualizamos el diccionario
        choice_dict[selected_value] += 1

        self._register_update(handle, choice_dict)

        # Devolvemos la opción
        return selected_value
        

    # model: {
    # (min, max) : ([values],[number_of_selections])])
    # }
    # La posición de cada value corresponde con la posición del
    # número de veces que fue escogido
    
    def discrete(self, min = 0, max = 10, handle=None):
        # En caso de no especificar la distribución,
        # devolvemos la de la clase superior
        if handle is None:
            return super().discrete(min, max, handle)

        # Obtenemos el diccionario asociado a esta distribución
        discrete_dict = self._get_model_params(handle, {})
        
        # Si los valores actuales no han sido vistos antes, los agregamos
        if not (min, max) in discrete_dict:
            discrete_dict[(min, max)] = ([], [])

        # Nos quedamos con una tupla de valor_elegido, cantidad_de_veces
        selections = discrete_dict[(min, max)]

        # En caso de que no se haya elegido algún valor hasta ahora
        if selections == ([], []):
            # Escogemos directamente como valor el centro del rango
            selected_value = int((min + max)/2)
        else:
            # En otro caso, verificamos si se desea que se devuelvan valores
            # cercanos o alejados a los ya devueltos
            if self._exploration:
                # En caso de que se deseen valores lejanos, verificamos algunos de los índices más elegidos
                # y escogemos uno al azar entre ello
                if len(selections[1]) == 1:
                    indice = 0
                else:
                    indices = best_indices(selections[1], random.randint(1, builtins.max(1, int(len(selections[1])/2))), True)
                    indice = indices[random.randint(0, len(indices)) - 1]
            else:
                # En caso de que se desee uno cercano a los elegidos, simplemente
                # escogemos unos al azar
                indice = random.randint(0, len(selections[0]) - 1)
            
            selected_value = selections[0][indice]

        
        # Calculamos el rango de alejamiento que debemos tener del valor elegido,
        # el cual estará dado por una proporción en base a alfa
        interval = max - min
        move_range = int(interval * (1 - self._alpha))

        # Calculamos el rango inferior con respecto a esta proporción
        value_negative = self._clamp(selected_value - move_range, min, max)

        # Si estamos en modo exploración, entonces elegimos un valor
        # entre el mínimo y el rango escogido
        if self._exploration:
            value_negative = random.randint(min, value_negative)
        # Si estamos en modo explotación, lo escogemos entre el rango
        # escogido y el valor
        else:
            value_negative = random.randint(value_negative, selected_value)

        # Realizamos el proceso opuesto con el rango superior de la proporción
        value_positive = self._clamp(selected_value + move_range, min, max)

        if self._exploration:
            value_positive = random.randint(value_positive, max)
        else:
            value_positive = random.randint(selected_value, value_positive)

        # Escogemos la azar uno de los dos valores como el valor final que vamos a devolver
        if random.uniform(0, 1) < 0.5:
            final_value = value_negative
        else:
            final_value = value_positive

        # Buscamos si hemos generado este valor previamente
        pos = find_pos(selections[0], final_value)

        # Si no lo hemos generado debemos agregarlo
        if pos == -1:
            selections[0].append(final_value)
            selections[1].append(1)
        # Si ya generamos el valor simplemente actualizamos
        else:
            selections[1][pos] += 1

        # Actualizamos el diccionario y lo agregamos al modelo
        discrete_dict[(min, max)] = selections
        
        self._register_update(handle, discrete_dict)

        # Retornamos el valor generado
        return final_value
        
    

    # model: {
    # (min, max) : [values_choosen]
    # }
    # Básicamente es una lista con todos los valores escogidos en ese rango
    def continuous(self, min=0, max=1, handle=None):
        if handle is None:
            return super().continuous(min, max, handle)
        
        # Obtenemos el diccionario asociado a esta distribución
        continuous_dict = self._get_model_params(handle, {})
        
        # Si los valores actuales no han sido vistos antes, los agregamos
        if not (min, max) in continuous_dict:
            continuous_dict[(min, max)] = []

        values_choosen = continuous_dict[(min, max)]

        # En caso de que no se haya elegido algún valor hasta ahora
        if values_choosen == []:
            # Escogemos directamente como valor el centro del rango
            selected_value = (min + max)/2
        else:
            # En otro caso, verificamos si se desea que se devuelvan valores
            # cercanos o alejados a los ya devueltos
            if self._exploration:
                # En caso de que se deseen valores lejanos, buscamos qué valores
                # tienen mayor cantidad de elementos cercanos en alfa a sí mismos,
                # en otras palabras, clubsterizamos
                if len(values_choosen) == 1:
                    indice = 0
                else:
                    selections = clubster_by_epsilon(values_choosen, (1 - self._alpha))
                    indices = best_indices(selections, random.randint(1, builtins.max(1, int(len(selections)/2))), True)
                    indice = indices[random.randint(0, len(indices)) - 1]
            else:
                # En caso de que se desee uno cercano a los elegidos, simplemente
                # escogemos unos al azar
                indice = random.randint(0, len(values_choosen) - 1)
            
            selected_value = values_choosen[indice]
        
        # Calculamos el rango de alejamiento que debemos tener del valor elegido,
        # el cual estará dado por una proporción en base a alfa
        interval = max - min
        move_range = interval * (1 - self._alpha)

        # Calculamos el rango inferior con respecto a esta proporción
        value_negative = self._clamp(selected_value - move_range, min, max)

        # Si estamos en modo exploración, entonces elegimos un valor
        # entre el mínimo y el rango escogido
        if self._exploration:
            value_negative = random.uniform(min, value_negative)
        # Si estamos en modo explotación, lo escogemos entre el rango
        # escogido y el valor
        else:
            value_negative = random.uniform(value_negative, selected_value)

        # Realizamos el proceso opuesto con el rango superior de la proporción
        value_positive = self._clamp(selected_value + move_range, min, max)

        if self._exploration:
            value_positive = random.uniform(value_positive, max)
        else:
            value_positive = random.uniform(selected_value, value_positive)

        # Escogemos la azar uno de los dos valores como el valor final que vamos a devolver
        if random.uniform(0, 1) < 0.5:
            final_value = value_negative
        else:
            final_value = value_positive

        #Agregamos este valor a los valores ya generados
        values_choosen.append(final_value)

        # Actualizamos el diccionario y lo agregamos al modelo
        continuous_dict[(min, max)] = values_choosen
        
        self._register_update(handle, continuous_dict)

        # Retornamos el valor generado
        return final_value


    # model: {
    # True: number_of_choosen,
    # False: number_of_choosen
    # } 
    # Básicamente es una lista con todos los valores escogidos en ese rango
    def boolean(self, handle=None):
        if handle is None:
            return super().boolean(handle)

        # Obtenemos los parámetros asociados a esta distribución
        boolean_dict = self._get_model_params(handle, {True: 0, False: 0})

        # Separamos la cantidad de veces que hemos elegido True
        # y la cantidad de veces que hemos escogido False
        true = boolean_dict[True]
        false = boolean_dict[False]

        # Vemos el total de elecciones que hemos realizado
        total = true + false

        # Si no hemos realizado ninguna hasta ahora,
        # escogemos al azar una de las dos
        if total == 0:
            if Bernoulli(0.5):
                true += 1
            else:
                false += 1
        # En caso de que hayamos seleccionado ya alguno de los dos
        else:
            # La idea será ver si se toma True o False con probabilidad
            # true/total, en caso de que se escoga True
            if Bernoulli(true/total):
                # Si estamos explorando
                if self._exploration:
                    # Y hemos escogido con menor cantidad los True,
                    # entonces hemos tomado la decisión certera
                    if true < false :
                        true += 1
                    # Si intentando explorar escogimos el que más
                    # se ha repetido, volvemos a calcular si tomarlo o no,
                    # esta vez a partir de alfa
                    else:
                        # En caso de que no sea escogido, volvemos al que
                        # menos veces ha sido escogido
                        if Bernoulli(1 - self._alpha):
                            false += 1
                        else:
                            # Si aún así fue escogido nuevamente, entonces
                            # será el valor devuelto
                            true += 1
                # Si estamos en modo explotación, utilizamos la misma lógica
                # para intentar escoger aquel que más veces se ha repetido
                else: 
                    if true > false :
                        true += 1
                    else:
                        if Bernoulli(1 - self._alpha):
                            false += 1
                        else:
                            true += 1
            else:
                # Aquí aplicamos la misma lógica, pero con False en vez de True
                if self._exploration:
                    if false < true:
                        false += 1
                    else:
                        if Bernoulli(1 - self._alpha):
                            true += 1
                        else:
                            false += 1
                else: 
                    if false > true :
                        false += 1
                    else:
                        if Bernoulli(1 - self._alpha):
                            true += 1
                        else:
                            false += 1
        
        #Una vez escogido el valor actualizamos el diccionario
        if true > boolean_dict[True]:
            final_value = True
            boolean_dict[True] = true
        else:
            final_value = False
            boolean_dict[False] = false
        
        # Actualizamos ahora el modelo
        self._register_update(handle, boolean_dict)

        # Retornamos el valor generado
        return final_value

        
    # model: {
    # [options] : [number_of_choosen]
    # }
    # Cada lista de opciones va a tener asociadas las veces
    # que fueron escogidas cada una de dichas opciones
    def categorical(self, options, handle=None):
        if handle is None:
            return super().categorical(options, handle)

        # Obtenemos el diccionario asociado a esta distribución
        categorical_dict = self._get_model_params(handle, {})

        if tuple(options) in categorical_dict:
            number_of_choosen = categorical_dict[tuple(options)]
        else:
            number_of_choosen = [0] * len(options)
        
        # Si solamente hay una opción, devolvemos directamente esta
        if len(options) == 1:
            indice = 0
        else:
            # Si hay más de una, en caso de estar explorando, seleccionamos una opción
            # entre aquellas que menos veces hemos escogido, en caso contrario, escogemos
            # una entre aquellas que más veces. El rango de selección está dado por alfa
            if self._exploration:
                indices = best_indices(number_of_choosen, random.randint(1, max(1, int(len(number_of_choosen)/2))), False)        
            else:
                indices = best_indices(number_of_choosen, random.randint(1, max(1, int(len(number_of_choosen)/2))), True)    
            selections_range = int(len(indices) * (1-self._alpha))
            indice = indices[random.randint(0, max(selections_range - 1, 0))]
        
        # Obtenemos la opción escogida
        selected_value = options[indice]

        # Actualizamos el diccionario
        number_of_choosen[indice] += 1
        categorical_dict[tuple(options)] = number_of_choosen

        self._register_update(handle, categorical_dict)

        # Devolvemos la opción
        return selected_value
        


def find_pos(list, value):
        for i in range(len(list)):
            if list[i] == value:
                return i
        return -1

def clubster_by_epsilon(list, epsilon):
    clubsters = [1] * len(list)

    for i in range(len(list)):
        for j in range(i + 1, len(list)):
            if list[j] < list[i] + epsilon:

                clubsters[i] += 1
                clubsters[j] += 1
                

    return clubsters

def Bernoulli(p : float):
        if p < 0 or p > 1:
            raise Exception("p parameters must be between 0 and 1")
        
        if random.random() < p:
            return True
        else:
            return False 
