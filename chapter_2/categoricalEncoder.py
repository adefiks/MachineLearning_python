# Definicja klasy CategoricalEncoder, skopiowana z prośby PR #9151.
# Uruchom tę komórkę nie próbując rozumieć jej zawartości (na razie).

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
import numpy as np


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Koduje cechy kategorialne w postaci macierzy numerycznej.
    Danymi wejściowymi dostarczanymi do tego transformatora powinna być macierz
    zawierająca liczby stałoprzecinkowe lub ciągi znaków, symbolizujące
    wartości przechowywane przez cechy kategorialne (dyskretne).
    Możemy kodować cechy za pomocą schematu "gorącojedynkowego" (jeden-z-K)
    (``encoding='onehot'``, domyślne rozwiązanie) lub przekształcać je do postaci
    liczb porządkowych (``encoding='ordinal'``).
    Tego typu kodowanie jest wymagane podczas dostarczania danych kategorialnych do wielu
    etymatorów modułu Scikit-Learn, mianowicie w modelach liniowych i maszynach
    SVM wykorzystujących standardowe jądra. Więcej informacji znajdziesz w:
    :ref:`User Guide <preprocessing_categorical_features>`.
    Parametry
    ----------
    encoding : ciąg znaków, 'onehot', 'onehot-dense' lub 'ordinal'
        Rodzaj stosowanego kodowania (domyślna wartość to 'onehot'):
        - 'onehot': koduje cechy za pomocą schematu "gorącojedynkowego" (jeden-z-K,
           bywa również nazywany kodowaniem 'sztucznym'). Zostaje utworzona kolumna
           binarna dla każdej kategorii, a zwracana jest macierz rzadka.
        - 'onehot-dense': to samo, co wartość 'onehot', ale zwraca macierz gęstą zamiast rzadkiej.
        - 'ordinal': koduje cechy w postaci liczb porządkowych. Uzyskujemy w ten sposób 
          pojedynczą kolumną zawierającą liczby stałoprzecinkowe (od 0 do n_kategorii - 1) 
          dla każdej cechy.
    categories : 'auto' lub lista list/tablic wartości.
        Kategorie (niepowtarzalne wartości) na każdą cechę:
        - 'auto' : Automatycznie określa kategorie za pomocą danych uczących. 
        - lista : ``categories[i]`` przechowuje kategorie oczekiwane w i-tej kolumnie.
          Przekazane kategorie zostają posortowanie przed zakodowaniem danych
          (użyte kategorie można przejrzeć w atrybucie ``categories_``).
    dtype : typ liczby, domyślnie np.float64
        Wymagany typ wartości wyjściowej.
    handle_unknown : 'error' (domyślnie) lub 'ignore'
        Za jego pomocą określamy, czy w przypadku obecności nieznanej cechy w czasie
        wykonywania transformacji ma być wyświetlany komunikat o błędzie (wartość
        domyślna) lub czy ma zostać zignorowana. Po wybraniu wartości 'ignore' 
        i natrafieniu na nieznaną kategorię w trakcie przekształceń, wygenerowane
        kolumny "gorącojedynkowe" dla tej cechy będą wypełnione zerami. 
        Ignorowanie nieznanych kategorii nie jest obsługiwane w parametrze
        ``encoding='ordinal'``.
    Atrybuty
    ----------
    categories_ : lista tablic
        Kategorie każdej cechy określone podczas uczenia. W przypadku ręcznego 
        wyznaczania kategorii znajdziemy tu listę posortowanych kategorii
        (w kolejności odpowiadającej wynikowi operacji 'transform').
    Przykłady
    --------
    Mając zbiór danych składający się z trzech cech i dwóch próbek pozwalamy koderowi
    znaleźć maksymalną wartość każdej cechy i przekształcić dane do postaci
    binarnego kodowania "gorącojedynkowego".
    >>> from sklearn.preprocessing import CategoricalEncoder
    >>> enc = CategoricalEncoder(handle_unknown='ignore')
    >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
    ... # doctest: +ELLIPSIS
    CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,
              encoding='onehot', handle_unknown='ignore')
    >>> enc.transform([[0, 1, 1], [1, 0, 4]]).toarray()
    array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]])
    Powiązane materiały
    --------
    sklearn.preprocessing.OneHotEncoder : przeprowadzana kodowanie "gorącojedynkowe"
      stałoprzecinkowych cech porządkowych. Klasa ``OneHotEncoder zakłada``, że cechy wejściowe
      przechowują wartości w zakresie ``[0, max(cecha)]`` zamiast korzystać z
      niepowtarzalnych wartości.
    sklearn.feature_extraction.DictVectorizer : przeprowadzana kodowanie "gorącojedynkowe"
      elementów słowanika (a także cech przechowujących ciągi znaków).
    sklearn.feature_extraction.FeatureHasher : przeprowadzana przybliżone kodowanie "gorącojedynkowe"
      elementów słownika lub ciągów znaków.
    """

    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Dopasowuje klasę CategoricalEncoder do danych wejściowych X.
        Parametry
        ----------
        X : tablicopodobny, postać [n_próbek, n_cech]
            Dane służące do określania kategorii każdej cechy.
        Zwraca
        -------
        self
        """

        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("Należy wybrać jedno z następujących kodowań: 'onehot', 'onehot-dense' "
                        "lub 'ordinal', wybrano %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("Należy wybrać jedną z następujących wartości parametru handle_unknown: 'error' lub "
                        "'ignore', wybrano %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("Wartość handle_unknown='ignore' nie jest obsługiwana przez parametr"
                             " encoding='ordinal'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Znaleziono nieznane kategorie {0} w kolumnie {1}"
                               " podczas dopasowywania".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        """Przekształca X za pomocą kodowania "gorącojedynkowego".
        Parametry
        ----------
        X : tablicopodobny, postać [n_próbek, n_cech]
            Kodowane dane.
        Zwraca
        -------
        X_out : macierz rzadka lub dwuwymiarowa tablica
            Przekształcone dane wejściowe.
        """
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Znaleziono nieznane kategorie {0} w kolumnie {1}"
                           " podczas przekształcania".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Wyznaczamy akceptowalną wartość rzędom sprawiającym problem i
                    # kontynuujemy. Rzędy te zostają oznaczone jako `X_mask` i zostaną
                    # później usunięte.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out
