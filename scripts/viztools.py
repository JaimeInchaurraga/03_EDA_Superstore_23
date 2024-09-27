import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def analyze_dataset(df):
    """
    Analyzes the dataset to provide insights about each column including:
    - Cardinality
    - Total unique values
    - Number of null values
    - Variable type (numerical continuous, numerical discrete, categorical, ordinal)

    Parameters:
    df (pd.DataFrame): DataFrame containing the data

    Returns:
    pd.DataFrame: DataFrame with the analysis of each column
    """
    
    def determine_variable_type(series):
        """
        Determines the variable type of a pandas series
        
        Parameters:
        series (pd.Series): The series to analyze
        
        Returns:
        str: The determined type of the variable
        """
        if pd.api.types.is_numeric_dtype(series):
            if series.nunique() / len(series) < 0.05:
                return 'Numerical Discrete'
            else:
                return 'Numerical Continuous'
        elif pd.api.types.is_categorical_dtype(series) or series.dtype == 'object':
            if series.nunique() / len(series) < 0.05:
                return 'Categorical'
            else:
                return 'Categorical High Cardinality'
        elif pd.api.types.is_datetime64_any_dtype(series):
            return 'Datetime'
        else:
            return 'Unknown'

    analysis = []

    for column in df.columns:
        unique_values = df[column].nunique()
        null_values = df[column].isnull().sum()
        variable_type = determine_variable_type(df[column])
        cardinality = (unique_values / len(df) * 100) if len(df) > 0 else np.nan
        
        analysis.append({
            'Column': column,
            'Unique Values': unique_values,
            'Null Values': null_values,
            'Variable Type': variable_type,
            'Cardinality': f"{cardinality:.2f}%" if not np.isnan(cardinality) else 'N/A'
        })

    analysis_df = pd.DataFrame(analysis)
    return analysis_df


def detect_outliers(data, factor=1.5):
    """
    Detecta outliers en una serie utilizando el rango intercuartílico (IQR).
    
    Parameters:
    data (pd.Series): Serie de datos a analizar
    factor (float): El multiplicador para el IQR para definir outliers (default es 1.5)
    
    Returns:
    pd.Series: Una serie de valores booleanos donde True indica un outlier
    """
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return (data < lower_bound) | (data > upper_bound)

def variabilidad(df, columns=None, high_dispersion_threshold=0.5, factor=1.5):
    if columns is None:
        # Seleccionar todas las columnas numéricas si no se especifican columnas
        columns = df.select_dtypes(include=np.number).columns.tolist()
    
    df_stats = df[columns].describe().T

    # Calcular el Coeficiente de Variación (CV)
    df_stats["CV"] = df_stats["std"] / df_stats["mean"]

    # Añadir una columna para indicar si la dispersión es alta
    df_stats["High Dispersion"] = df_stats["CV"] > high_dispersion_threshold

    # Detectar outliers en cada columna
    outlier_flags = df[columns].apply(detect_outliers, factor=factor)
    
    # Calcular estadísticas sin outliers
    df_no_outliers = df[~outlier_flags.any(axis=1)][columns]
    df_stats_no_outliers = df_no_outliers.describe().T
    
    # Calcular el Coeficiente de Variación (CV) sin outliers
    df_stats_no_outliers["CV"] = df_stats_no_outliers["std"] / df_stats_no_outliers["mean"]
    
    # Añadir una columna para indicar si la dispersión es alta sin outliers
    df_stats_no_outliers["High Dispersion"] = df_stats_no_outliers["CV"] > high_dispersion_threshold
    
    # Seleccionar las columnas de interés y combinar con los datos originales
    df_stats = df_stats[["mean", "std", "CV", "High Dispersion"]]
    df_stats_no_outliers = df_stats_no_outliers[["mean", "std", "CV", "High Dispersion"]]
    
    return df_stats, df_stats_no_outliers


# Ejemplo de uso:
# df = pd.read_csv('your_dataset.csv')
# df_var_with_outliers, df_var_without_outliers = variabilidad(df, columns=['Sales', 'Profit'])
# print("Con Outliers:")
# print(df_var_with_outliers)
# print("\nSin Outliers:")
# print(df_var_without_outliers)



# Gráfico de barras para mostrar distribución de categorías
def pinta_distribucion_categoricas(df, columnas_categoricas, relativa=False, mostrar_valores=False):
    num_columnas = len(columnas_categoricas)
    num_filas = (num_columnas // 2) + (num_columnas % 2)

    fig, axes = plt.subplots(num_filas, 2, figsize=(15, 5 * num_filas))
    axes = axes.flatten() 

    for i, col in enumerate(columnas_categoricas):
        ax = axes[i]
        if relativa:
            total = df[col].value_counts().sum()
            serie = df[col].value_counts().apply(lambda x: x / total)
            sns.barplot(x=serie.index, y=serie, ax=ax, palette='viridis', hue = serie.index, legend = False)
            ax.set_ylabel('Frecuencia Relativa')
        else:
            serie = df[col].value_counts()
            sns.barplot(x=serie.index, y=serie, ax=ax, palette='viridis', hue = serie.index, legend = False)
            ax.set_ylabel('Frecuencia')

        ax.set_title(f'Distribución de {col}')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)

        if mostrar_valores:
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    for j in range(i + 1, num_filas * 2):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def plot_categorical_relationship_fin(df, cat_col1, cat_col2, relative_freq=False, show_values=False, size_group = 5):
    # Prepara los datos
    count_data = df.groupby([cat_col1, cat_col2]).size().reset_index(name='count')
    total_counts = df[cat_col1].value_counts()
    
    # Convierte a frecuencias relativas si se solicita
    if relative_freq:
        count_data['count'] = count_data.apply(lambda x: x['count'] / total_counts[x[cat_col1]], axis=1)

    # Si hay más de size_group categorías en cat_col1, las divide en grupos de size_group
    unique_categories = df[cat_col1].unique()
    if len(unique_categories) > size_group:
        num_plots = int(np.ceil(len(unique_categories) / size_group))

        for i in range(num_plots):
            # Selecciona un subconjunto de categorías para cada gráfico
            categories_subset = unique_categories[i * size_group:(i + 1) * size_group]
            data_subset = count_data[count_data[cat_col1].isin(categories_subset)]

            # Crea el gráfico
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=cat_col1, y='count', hue=cat_col2, data=data_subset, order=categories_subset)

            # Añade títulos y etiquetas
            plt.title(f'Relación entre {cat_col1} y {cat_col2} - Grupo {i + 1}')
            plt.xlabel(cat_col1)
            plt.ylabel('Frecuencia' if relative_freq else 'Conteo')
            plt.xticks(rotation=45)

            # Mostrar valores en el gráfico
            if show_values:
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', fontsize=10, color='black', xytext=(0, size_group),
                                textcoords='offset points')

            # Muestra el gráfico
            plt.show()
    else:
        # Crea el gráfico para menos de size_group categorías
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=cat_col1, y='count', hue=cat_col2, data=count_data)

        # Añade títulos y etiquetas
        plt.title(f'Relación entre {cat_col1} y {cat_col2}')
        plt.xlabel(cat_col1)
        plt.ylabel('Frecuencia' if relative_freq else 'Conteo')
        plt.xticks(rotation=45)

        # Mostrar valores en el gráfico
        if show_values:
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=10, color='black', xytext=(0, size_group),
                            textcoords='offset points')

        # Muestra el gráfico
        plt.show()


# Gráfico de barras apiladas para ver la composición de una categoría dentro de otra
def plot_categorical_composition(data, var1, var2):
    """
    Genera un gráfico de barras apiladas para mostrar la composición de una variable categórica (var2) 
    dentro de otra (var1), ordenando por el total de ocurrencias de var1.

    Parámetros:
    data (pd.DataFrame): DataFrame con los datos.
    var1 (str): Nombre de la primera variable categórica (por ejemplo, 'State').
    var2 (str): Nombre de la segunda variable categórica (por ejemplo, 'Order Profitable?') que muestra la composición.
    """
    
    # Crear tabla de proporciones de la variable categórica var1 y var2
    counts = pd.crosstab(data[var1], data[var2])

    # Calcular el total de pedidos sumando las categorías de var2 y ordenar por ese total
    counts['Total Orders'] = counts.sum(axis=1)
    counts_sorted_desc = counts.sort_values('Total Orders', ascending=True)

    # Crear proporciones para apilado con las categorías ordenadas correctamente
    proportions_sorted_desc = counts_sorted_desc.drop(columns='Total Orders').div(counts_sorted_desc['Total Orders'], axis=0)

    # Generar una paleta de colores dependiendo del número de categorías en var2
    num_categories = len(counts.columns) - 1  # Excluir 'Total Orders'
    colors = sns.color_palette("Set3", num_categories)  # Paleta de colores con un número dinámico de colores

    # Crear el gráfico de barras apiladas con colores dinámicos
    fig, ax = plt.subplots(figsize=(10, 12))

    # Crear el gráfico apilado con las categorías ordenadas correctamente
    proportions_sorted_desc.plot(kind='barh', stacked=True, color=colors, ax=ax, edgecolor='none')

    # Añadir los números de pedidos en cada barra
    for i, (state, row) in enumerate(counts_sorted_desc.iterrows()):
        total_orders = row['Total Orders']
        ax.text(1.05, i, str(int(total_orders)), va='center', color='black', fontweight='bold')
        for j, value in enumerate(row[:-1]):  # Excluir 'Total Orders' de las iteraciones
            if value > 0:
                x_position = row.cumsum()[j] - (value / 2)
                ax.text(x_position / total_orders, i, str(int(value)), va='center', color='black')

    # Añadir etiquetas y título
    ax.set_xlabel('Proporción')
    ax.set_ylabel(var1)
    ax.set_title(f'Proporción de {var2} dentro de {var1} (ordenado por total)')

    # Ajustar los márgenes para dejar más espacio en blanco en el margen derecho
    plt.subplots_adjust(right=0.8)

    # Mostrar el gráfico
    plt.show()




def plot_categorical_numerical_relationship(df, categorical_col, numerical_col, show_values=False, measure='mean'):
    # Calcula la medida de tendencia central (mean o median)
    if measure == 'median':
        grouped_data = df.groupby(categorical_col)[numerical_col].median()
    else:
        # Por defecto, usa la media
        grouped_data = df.groupby(categorical_col)[numerical_col].mean()

    # Ordena los valores
    grouped_data = grouped_data.sort_values(ascending=False)

    # Si hay más de 5 categorías, las divide en grupos de 5
    if grouped_data.shape[0] > 5:
        unique_categories = grouped_data.index.unique()
        num_plots = int(np.ceil(len(unique_categories) / 5))

        for i in range(num_plots):
            # Selecciona un subconjunto de categorías para cada gráfico
            categories_subset = unique_categories[i * 5:(i + 1) * 5]
            data_subset = grouped_data.loc[categories_subset]

            # Crea el gráfico
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=data_subset.index, y=data_subset.values)

            # Añade títulos y etiquetas
            plt.title(f'Relación entre {categorical_col} y {numerical_col} - Grupo {i + 1}')
            plt.xlabel(categorical_col)
            plt.ylabel(f'{measure.capitalize()} de {numerical_col}')
            plt.xticks(rotation=45)

            # Mostrar valores en el gráfico
            if show_values:
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                                textcoords='offset points')

            # Muestra el gráfico
            plt.show()
    else:
        # Crea el gráfico para menos de 5 categorías
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=grouped_data.index, y=grouped_data.values)

        # Añade títulos y etiquetas
        plt.title(f'Relación entre {categorical_col} y {numerical_col}')
        plt.xlabel(categorical_col)
        plt.ylabel(f'{measure.capitalize()} de {numerical_col}')
        plt.xticks(rotation=45)

        # Mostrar valores en el gráfico
        if show_values:
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                            textcoords='offset points')

        # Muestra el gráfico
        plt.show()


def plot_combined_graphs(df, columns, whisker_width=1.5, bins = None):
    num_cols = len(columns)
    if num_cols:
        
        fig, axes = plt.subplots(num_cols, 2, figsize=(12, 5 * num_cols))
        print(axes.shape)

        for i, column in enumerate(columns):
            if df[column].dtype in ['int64', 'float64']:
                # Histograma y KDE
                sns.histplot(df[column], kde=True, ax=axes[i,0] if num_cols > 1 else axes[0], bins= "auto" if not bins else bins)
                if num_cols > 1:
                    axes[i,0].set_title(f'Histograma y KDE de {column}')
                else:
                    axes[0].set_title(f'Histograma y KDE de {column}')

                # Boxplot
                sns.boxplot(x=df[column], ax=axes[i,1] if num_cols > 1 else axes[1], whis=whisker_width)
                if num_cols > 1:
                    axes[i,1].set_title(f'Boxplot de {column}')
                else:
                    axes[1].set_title(f'Boxplot de {column}')

        plt.tight_layout()
        plt.show()

def plot_multiple_boxplots(df, columns, dim_matriz_visual = 2):
    num_cols = len(columns)
    num_rows = num_cols // dim_matriz_visual + num_cols % dim_matriz_visual
    fig, axes = plt.subplots(num_rows, dim_matriz_visual, figsize=(12, 6 * num_rows))
    axes = axes.flatten()

    for i, column in enumerate(columns):
        if df[column].dtype in ['int64', 'float64']:
            sns.boxplot(data=df, x=column, ax=axes[i])
            axes[i].set_title(column)

    # Ocultar ejes vacíos
    for j in range(i+1, num_rows * 2):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# Gráfica para boxplots uniendo una variable categorica y otra numérica. Se pueden usar violines y muestra la densidad de cada variable teniendo en cuenta el número de valores totales por categoría (suuuper útil)

def plot_grouped_plots(df, cat_col, num_col, plot_type='boxplot', median_color='red'):
    """
    Genera gráficos de cajas (boxplots) o violines (violin plots) con la opción de resaltar la mediana.
    
    Parámetros:
    - df (DataFrame): El DataFrame que contiene los datos.
    - cat_col (str): El nombre de la columna categórica.
    - num_col (str): El nombre de la columna numérica.
    - plot_type (str): El tipo de gráfico, puede ser 'boxplot' o 'violin'. Por defecto, es 'boxplot'.
    - median_color (str): El color para resaltar la mediana. Por defecto, es 'red'.
    """
    
    # Obtener las categorías únicas y su número
    unique_cats = df[cat_col].unique()
    num_cats = len(unique_cats)
    group_size = 5  # Cantidad de categorías por gráfico

    # Bucle para dividir las categorías en grupos más pequeños
    for i in range(0, num_cats, group_size):
        subset_cats = unique_cats[i:i+group_size]
        subset_df = df[df[cat_col].isin(subset_cats)]
        
        plt.figure(figsize=(10, 6))
        
        if plot_type == 'boxplot':
            # Crear el boxplot y resaltar la mediana
            sns.boxplot(x=cat_col, y=num_col, data=subset_df,
                        medianprops=dict(color=median_color, linewidth=2))
        elif plot_type == 'violin':
            # Crear el violin plot con densidad ajustada al tamaño de los grupos
            sns.violinplot(x=cat_col, y=num_col, data=subset_df, inner="quartile", cut=0, density_norm="count")
            # Añadir la mediana con un boxplot reducido
            sns.boxplot(x=cat_col, y=num_col, data=subset_df, whis=[0, 100], width=0.1, showfliers=False,
                        medianprops=dict(color=median_color, linewidth=2))
        
        plt.title(f'{plot_type.capitalize()} of {num_col} for {cat_col} (Group {i//group_size + 1})')
        plt.xticks(rotation=45)
        plt.show()


def plot_histo_den(df, columns):
    num_cols = len(columns)
    num_rows = num_cols // 2 + num_cols % 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(12, 6 * num_rows))
    axes = axes.flatten()

    for i, column in enumerate(columns):
        if df[column].dtype in ['int64', 'float64']:
            sns.histplot(df[column], kde=True, ax=axes[i])
            axes[i].set_title(f'Histograma y KDE de {column}')

    # Ocultar ejes vacíos
    for j in range(i + 1, num_rows * 2):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()



def plot_grouped_histograms(df, cat_col, num_col, group_size):
    unique_cats = df[cat_col].unique()
    num_cats = len(unique_cats)

    for i in range(0, num_cats, group_size):
        subset_cats = unique_cats[i:i+group_size]
        subset_df = df[df[cat_col].isin(subset_cats)]
        
        plt.figure(figsize=(10, 6))
        for cat in subset_cats:
            sns.histplot(subset_df[subset_df[cat_col] == cat][num_col], kde=True, label=str(cat))
        
        plt.title(f'Histograms of {num_col} for {cat_col} (Group {i//group_size + 1})')
        plt.xlabel(num_col)
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

def plot_histograms_huge_data(df, columns, limit, bins_large=100):
    """
    Crea histogramas para cada columna seleccionada en el DataFrame.
    Cada columna tendrá dos gráficos: uno para los valores menores o iguales al límite y otro para los valores mayores al límite.
    
    Parameters:
    df (pd.DataFrame): El DataFrame que contiene los datos.
    columns (list of str): La lista de columnas sobre las que se crearán los histogramas.
    limit (float): El valor límite para dividir los datos en dos gráficos.
    bins_large (int): El número de bins para el histograma de valores mayores al límite.
    """
    num_cols = len(columns)
    
    # Crear la figura y los subplots
    fig, axes = plt.subplots(nrows=num_cols, ncols=2, figsize=(14, 6 * num_cols))
    
    # Asegurar que axes sea una lista de listas en caso de una sola columna
    if num_cols == 1:
        axes = [axes]
    
    for idx, column in enumerate(columns):
        # Filtrar valores menores o iguales al límite
        filtered_df_small = df[df[column] <= limit]
        # Crear el histograma para valores <= límite
        sns.histplot(filtered_df_small[column], bins=50, kde=True, ax=axes[idx][0])
        axes[idx][0].set_title(f"Histograma de {column} <= {limit}")
        axes[idx][0].set_xlabel(column)
        axes[idx][0].set_ylabel("Frecuencia")
        
        # Filtrar valores mayores al límite
        filtered_df_large = df[df[column] > limit]
        # Crear el histograma para valores > límite
        sns.histplot(filtered_df_large[column], bins=bins_large, kde=True, ax=axes[idx][1])
        axes[idx][1].set_title(f"Histograma de {column} > {limit}")
        axes[idx][1].set_xlabel(column)
        axes[idx][1].set_ylabel("Frecuencia")
        axes[idx][1].set_xlim(limit, filtered_df_large[column].max())  # Ajustar el límite del eje x
    
    # Ajustar el layout para evitar solapamientos
    plt.tight_layout()
    
    # Mostrar la figura con los subplots
    plt.show()

# Gráfica estilo lollipop, buena para ciertos casos de una varuable categorica y otra numerica

def lollipop_plot(df, num_col, cat_col, agg_func='mean'):
    """
    Función para crear un gráfico lollipop horizontal que muestra la relación entre una variable categórica y una numérica.
    
    Parámetros:
    - df (pd.DataFrame): El DataFrame que contiene los datos.
    - num_col (str): El nombre de la columna numérica.
    - cat_col (str): El nombre de la columna categórica.
    - agg_func (str): La medida estadística a usar ('mean', 'median', 'sum', etc.). Por defecto es 'mean'.
    
    Retorna:
    - Un gráfico lollipop horizontal.
    """
    
    # Validar la medida estadística seleccionada
    if agg_func not in ['mean', 'median', 'sum', 'min', 'max']:
        raise ValueError("La medida debe ser una de las siguientes: 'mean', 'median', 'sum', 'min', 'max'")
    
    # Agrupar los datos según la categoría y calcular la medida numérica seleccionada
    if agg_func == 'mean':
        df_agg = df.groupby(cat_col)[num_col].mean().reset_index()
    elif agg_func == 'median':
        df_agg = df.groupby(cat_col)[num_col].median().reset_index()
    elif agg_func == 'sum':
        df_agg = df.groupby(cat_col)[num_col].sum().reset_index()
    elif agg_func == 'min':
        df_agg = df.groupby(cat_col)[num_col].min().reset_index()
    elif agg_func == 'max':
        df_agg = df.groupby(cat_col)[num_col].max().reset_index()
    
    # Reordenar según los valores de la columna numérica
    ordered_df = df_agg.sort_values(by=num_col)
    my_range = range(1, len(ordered_df.index) + 1)

    # Crear el lollipop plot horizontal
    plt.figure(figsize=(10, 6))
    plt.hlines(y=my_range, xmin=0, xmax=ordered_df[num_col], color='skyblue')
    plt.plot(ordered_df[num_col], my_range, "o", color='skyblue')

    # Añadir títulos y nombres de ejes
    plt.yticks(my_range, ordered_df[cat_col])
    plt.title(f'{agg_func.capitalize()} de {num_col} por {cat_col}', loc='center')
    plt.xlabel(f'{agg_func.capitalize()} de {num_col}')
    plt.ylabel(cat_col)

    # Mostrar la gráfica
    plt.show()




def grafico_dispersion_con_correlacion(df, columna_x, columna_y, tamano_puntos=50, mostrar_correlacion=False):
    """
    Crea un diagrama de dispersión entre dos columnas y opcionalmente muestra la correlación.

    Args:
    df (pandas.DataFrame): DataFrame que contiene los datos.
    columna_x (str): Nombre de la columna para el eje X.
    columna_y (str): Nombre de la columna para el eje Y.
    tamano_puntos (int, opcional): Tamaño de los puntos en el gráfico. Por defecto es 50.
    mostrar_correlacion (bool, opcional): Si es True, muestra la correlación en el gráfico. Por defecto es False.
    """

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=columna_x, y=columna_y, s=tamano_puntos)

    if mostrar_correlacion:
        correlacion = df[[columna_x, columna_y]].corr().iloc[0, 1]
        plt.title(f'Diagrama de Dispersión con Correlación: {correlacion:.2f}')
    else:
        plt.title('Diagrama de Dispersión')

    plt.xlabel(columna_x)
    plt.ylabel(columna_y)
    plt.grid(True)
    plt.show()


def bubble_plot(df, col_x, col_y, col_size, scale = 1000):
    """
    Crea un scatter plot usando dos columnas para los ejes X e Y,
    y una tercera columna para determinar el tamaño de los puntos.

    Args:
    df (pd.DataFrame): DataFrame de pandas.
    col_x (str): Nombre de la columna para el eje X.
    col_y (str): Nombre de la columna para el eje Y.
    col_size (str): Nombre de la columna para determinar el tamaño de los puntos.
    """

    # Asegúrate de que los valores de tamaño sean positivos
    sizes = (df[col_size] - df[col_size].min() + 1)/scale

    plt.scatter(df[col_x], df[col_y], s=sizes)
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.title(f'Burbujas de {col_x} vs {col_y} con Tamaño basado en {col_size}')
    plt.show()

def plot_stacked_bar_periodos(df, time_column, value_column, period='months'):
    """
    Plot a stacked bar chart based on the specified time period.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data
    time_column (str): Column name for the time series data
    value_column (str): Column name for the value to plot on the y-axis
    period (str): Time period to aggregate by ('months', 'quarters', 'years')

    """
    df_copy = df.copy()
    
    if period == 'months':
        df_copy['Period'] = df_copy[time_column].dt.strftime('%B')
        order = ['January', 'February', 'March', 'April', 'May', 'June', 
                 'July', 'August', 'September', 'October', 'November', 'December']
    elif period == 'quarters':
        df_copy['Period'] = df_copy[time_column].dt.to_period('Q').astype(str)
        order = sorted(df_copy['Period'].unique())
    elif period == 'years':
        df_copy['Period'] = df_copy[time_column].dt.year.astype(str)
        order = sorted(df_copy['Period'].unique())
    else:
        raise ValueError("Period must be one of 'months', 'quarters', 'years'")
    
    df_copy['Year'] = df_copy[time_column].dt.year
    pivot_table = df_copy.pivot_table(index='Period', columns='Year', values=value_column, aggfunc='sum', fill_value=0)
    pivot_table = pivot_table.reindex(order)
    
    ax = pivot_table.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='Oranges')
    
    plt.title(f'Stacked Bar Chart - {period.capitalize()}')
    plt.xlabel('Period')
    plt.ylabel('Sales')
    plt.legend(title='Year')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_order_profitability_by_discount(df, discount_col, profitability_col):
    """
    Esta función crea un gráfico de barras apiladas para mostrar la rentabilidad de los pedidos
    según el nivel de descuento, y añade el número total de pedidos en la parte superior de cada barra.
    
    Parámetros:
    - df (DataFrame): El DataFrame que contiene los datos.
    - discount_col (str): El nombre de la columna que contiene los niveles de descuento.
    - profitability_col (str): El nombre de la columna que contiene la información de si el pedido fue rentable o no.
    """
    # Agrupar los datos por nivel de descuento y si fue rentable o no, y contar el número de pedidos
    counts = df.groupby([discount_col, profitability_col]).size().unstack(fill_value=0)

    # Calcular el porcentaje de cada categoría dentro de cada nivel de descuento
    percentages = counts.div(counts.sum(axis=1), axis=0)

    # Crear el gráfico de barras apiladas
    ax = percentages.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='Set1')

    # Añadir el número total de pedidos en la parte superior de cada barra
    totals = counts.sum(axis=1)
    for i, total in enumerate(totals):
        ax.text(i, 1.02, f'{total}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Añadir etiquetas y título
    plt.xlabel('Discount Level')
    plt.ylabel('Percentage of Orders')
    plt.title('Order Profitability by Discount Level')
    plt.legend(title='Order Profitable?', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Ajustar el diseño
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()