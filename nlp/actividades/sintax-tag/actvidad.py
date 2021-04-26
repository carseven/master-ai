import pandas as pd


class Palabra:
    '''
    Clase para guardar el token y la etiqueta de una palabra de un corpus
    '''

    def __init__(self, token: str, tag: str):
        '''
        Constructor de la clase

        token : str
            Token de la palabra

        tag : str
            Etiqueta de la palabra
        '''
        self._token = token
        self._tag = tag

    def Token(self):
        '''
        Método para acceder al token de la palabra
        '''
        return self._token

    def Tag(self):
        '''
        Método par acceder a la etiqueta de la palabra
        '''
        return self._tag


class HMMBigrama:
    '''
    Clase para obtener las matrices de probabilidad HMM Bigrama a partir de un
    corpus
    '''

    def __init__(self, corpus: list[list[Palabra]]):
        '''
        Constructor de la clase para calcular el Modelo Oculto de Markov
        Bigrama
        '''
        self._corpus = corpus
        self._estados = dict()
        self._tokens = dict()
        self._q0 = 'q0'
        self._qF = 'qF'

        self._prob_trans = pd.DataFrame()
        self._prob_obs = pd.DataFrame()

    def Corpus(self):
        return self._corpus.copy()

    def EstadoInicial(self):
        return self._q0

    def EstadoFinal(self):
        return self._qF

    def _ProcesarCorpus(self):
        '''
        Método para contar el número de ocurrencias de estados y tokens
        '''
        for oracion in self._corpus:
            for palabra in oracion:

                ##################################################
                ########## Aquí debes incluir tu código ##########
                ##################################################

                # Contar ocurrencias tokens
                if palabra.Token() not in self._tokens:
                    self._tokens[palabra.Token()] = 0
                self._tokens[palabra.Token()] += 1

                # Contar ocurrencias estados
                if palabra.Tag() not in self._estados:
                    self._estados[palabra.Tag()] = 0
                self._estados[palabra.Tag()] += 1

    def Estados(self, incluir_inicial: bool = False,
                incluir_final: bool = False):
        '''
        Devuelve los estados del bigrama en base al corpus proporcionado al
        constructor

        incluir_inicial : bool (False)
            Flag para indicar si se quiere recuperar el estado inicial

        incluir_final : bool (False)
            Flag para indicar si se quiere recuperar el estado final

        return
            Diccionario de estados con el número de ocurrencias de cada estado
            en el corpus
        '''

        if len(self._estados) == 0:
            self._ProcesarCorpus()

        copia_estados = dict()
        if incluir_inicial:
            # Hay tantos estados como oraciones en el corpus
            copia_estados[self._q0] = len(self._corpus)

        copia_estados.update(self._estados)

        if incluir_final:
            # Hay tantos estados como oraciones en el corpus
            copia_estados[self._qF] = len(self._corpus)

        return copia_estados

    def Tokens(self):
        '''
        Devuelve los tokens del bigrama en base al corpus proporcionado al
        constructor

        return
            Diccionario de tokens con el número de ocurrencias de cada token
            en el corpus
        '''

        if len(self._tokens) == 0:
            self._ProcesarCorpus()

        return self._tokens.copy()

    def ProbabilidadesDeTransicion(self):
        '''
        Método para calcular las probabilidades de transición bigrama
        a partir del corpus proporcionado a la clase
        '''

        # Si ya se ha calculado se devuelve
        if len(self._prob_trans) != 0:
            return self._prob_trans.copy()

        '''
        En esta parte del código se calcula el número de
        transiciones bigrama, es decir, en el diccionario
        'contador_transiciones' se almacenarán los contadores
        de las transiciones t-1 -> t

        Las claves del diccionario serán los estados de partida
        mientras que los valores de cada clave serán los estados
        de destino y el número de veces que transitan a cada estado
        '''
        q0 = self._q0
        qF = self._qF
        contador_transiciones = {q0: dict()}

        ##################################################
        ########## Aquí debes incluir tu código ##########
        ##################################################

        # Inicializamos estados
        contador_transiciones.update({t_1: dict()
                                      for t_1 in self.Estados().keys()})

        for oracion in self._corpus:
            # Inicializar estado anterior en cada nueva oración.
            t_1 = str()
            for palabra in oracion:
                # INIT -> Primera palabra de la oración.
                if not t_1:
                    contador_transiciones['q0'][palabra.Tag()
                                                ] = contador_transiciones['q0'].get(palabra.Tag(), 0) + 1
                    t_1 = palabra.Tag()
                    continue

                # Contar transiciones
                if palabra.Tag() not in contador_transiciones[t_1]:
                    contador_transiciones[t_1][palabra.Tag()] = 1
                else:
                    contador_transiciones[t_1][palabra.Tag()] += 1

                # Actualizar estado anterior
                t_1 = palabra.Tag()

            # Ultima palabra de la oración -> FINAL
            contador_transiciones[oracion[-1].Tag()]['qF'
                                                     ] = contador_transiciones[oracion[-1].Tag()].get('qF', 0) + 1

        '''
        Cálculo de la tabla de probabilidades de transición.

        Se calculan ahora las probabilidades de transición
        siguiendo la relación: P(T|T-1) = C(T-1, T) / C(T-1).

        En 'contador_transiciones' se han acumulado la coincidencias C(T-1, T)
        y en 'estados' se tiene disponible C(T-1) por lo que es posible
        calcular la tabla de probabilidades de transiciones con estos
        elementos.
        '''
        tags_estados_iniciales = list(
            self.Estados(incluir_inicial=True).keys())
        tags_estados_finales = list(self.Estados(incluir_final=True).keys())
        estados_totales = self.Estados(
            incluir_inicial=True, incluir_final=True)

        prob_trans = {qt_1: {qt: 0 for qt in tags_estados_finales}
                      for qt_1 in tags_estados_iniciales}
        for key, value in prob_trans.items():
            print(f"{key} -> {value}\n")
        ##################################################
        ########## Aquí debes incluir tu código ##########
        ##################################################
        for t_1 in contador_transiciones.keys():
            for t in contador_transiciones[t_1].keys():
                prob_trans[t_1][t] = contador_transiciones[t_1][t] / \
                    estados_totales[t_1]
        for t_1, values in prob_trans.items():
            print(t_1, '\n')
            for t, value in values.items():
                if value > 0:
                    print(f"{t} {value}")
            print('\n')
        return self._prob_trans.copy()

    def ProbabilidadesDeEmision(self):
        '''
        Método para calcular las probabilidades de emisión
        a partir del corpus proporcionado a la clase
        '''

        if len(self._prob_obs) != 0:
            return self._prob_obs.copy()

        '''
        En esta parte del código se calculan el número de
        ocurrencias de la palabra Wi para la etiqueta Ti
        '''
        estados = self.Estados()
        contador_observaciones = {key: dict() for key in estados.keys()}

        ##################################################
        ########## Aquí debes incluir tu código ##########
        ##################################################

        '''
        Cálculo de la tabla de probabilidades de emisión.

        Se calculan ahora las probabilidades de emisión
        siguiendo la relación: P(Wi|Ti) = C(Ti,Wi) / C(Ti).

        En 'contador_observaciones' se han acumulado la coincidencias C(Ti, Wi)
        y en 'estados' se tiene disponible C(Ti) por lo que es posible
        calcular la tabla de probabilidad de emisión con estos elementos.
        '''
        tokens = self.Tokens()
        prob_obs = {Ti: {Wi: 0 for Wi in tokens} for Ti in estados}

        ##################################################
        ########## Aquí debes incluir tu código ##########
        ##################################################

        return self._prob_obs


def main():
    archivo = open('mia07_t3_tra_Corpus-tagged.txt', "r")

    corpus = list()
    oracion_actual = list()

    for entrada in archivo.readlines():
        entrada = entrada.split()
        if len(entrada) == 0:
            # Puede ser la primera oración del documento
            # O que termina la oración
            if len(oracion_actual) > 0:
                # Fin de la oración
                corpus.append(oracion_actual)
            oracion_actual = list()
            continue

        elif entrada[0] == '<doc':
            # Inicio de documento. No se hace nada
            continue

        elif entrada[0] == '</doc>':
            # Fin del documento. No se hace nada
            continue

        oracion_actual.append(Palabra(token=entrada[0], tag=entrada[2]))

    archivo.close()

    hmmbigrama = HMMBigrama(corpus)

    # print(hmmbigrama.Tokens())
    # print(len(hmmbigrama.Tokens()))
    # print(hmmbigrama.Estados())
    # print(len(hmmbigrama.Estados()))
    hmmbigrama.ProbabilidadesDeTransicion()


if __name__ == "__main__":
    main()
