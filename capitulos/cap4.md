# Detecção de pontos de interesse

Neste capítulo iremos tratar sobre pontos característicos informativos em imagens, pontos de interesse. Consiste em um ponto expressivo em textura e seu conceito é baseado na intuição de que a identificação de pontos marcantes na imagem poderia ser útil para representação de imagem e objetos de forma invariante em outras imagens semelhantes da mesma cena e objeto.  Também serão demonstrados alguns algoritmos que podem ser utilizados para a detecção destes pontos, também conhecidos como Keypoints.


**Os códigos deste capítulo estão disponíveis no Google Colab. Caso desejem realizar alterações no código é necessário realizar uma cópia do arquivo.**

## Prepadando o ambiente

Neste capítulo iremos utilizar o google colab com as seguintes versões das bibliotecas:
- Python 3.8.5
- Bibliotecas (**OpenCV**, **Numpy**).


## Entendendo o Problema
A principal tarefa de um detector de ponto-chave ou ponto de interesse é atribuir uma pontuação de saliência a cada pixel de uma imagem. Esta pontuação é então usada para selecionar um conjunto de pixels que apresentam as seguintes propriedades:
1. **Repetibilidade**: Os pixels selecionados devem permanecer estáveis 
2. **Distinção**: A vizinhança em torno de cada ponto-chave deve apresentar um padrão de intensidade com fortes variações;
3. **Localidade**: Os recursos devem ser uma função de informações locais;
4. **Localidade em precisão**: O processo de localização deve ser sujeito a erros menores em relação à escala e forma;
5. **Eficiente**: Baixo tempo para processar

Uma forma de pensar sobre este tipo de representação é imaginar que os pontos correspondem a peças de um quebra-cabeça. Enquanto algumas peças são fáceis de serem identificadas, como olhos de um personagem, partes de um objeto muito diferente do restante da cena e até mesmo partes da roupa que o personagem está utilizando, algumas partes podem ser difíceis de serem identificadas. Um exemplo deste segundo tipo de ponto poderia ser o fundo em que o desenho está inserido, uma parte lisa da roupa do personagem que não te dá muitas informações sobre de onde poderia ser esta peça, entre outros exemplos. Desta forma, os pontos chaves para você começar a montar seu quebra cabeça seriam as peças que mais se destacam. E é esta mesma ideia que usaremos para encontrar os keypoints em nossa imagem.

Outro exemplo também é a perícia de escrita. Imagine que estamos investigando se um suspeito X escreveu ou não a carta encontrada na cena do crime. Uma forma de realizar esta análise poderia ser pegar um texto que foi escrito pelo suspeito e procurar os principais traços que diferenciam sua escrita. Desta forma, para verificar se as assinaturas batem, o que teríamos que fazer é apenas comparar se estes traços principais se assemelham nas duas cartas ou não, sem a necessidade de comparar todas as letras.

Quando se trata de algoritmos que realizam a identificação destes pontos existem diversas abordagens. Iremos descrever as principais:

### Harris Corner
O Harris Corner Detector é um operador de detecção de cantos comumente usado em algoritmos de visão computacional para extrair cantos e inferir características de uma imagem. Foi introduzido pela primeira vez por Chris Harris e Mike Stephens em 1988 com o aprimoramento do detector de canto de Moravec. Em comparação com o anterior, o detector de canto de Harris leva o diferencial da pontuação de canto em consideração com referência à direção diretamente, em vez de usar remendos de deslocamento para cada ângulo de 45 graus, e provou ser mais preciso na distinção entre bordas e cantos. Desde então, ele foi aprimorado e adotado em muitos algoritmos para pré-processar imagens para aplicações subsequentes.

#### Mas o que seria um Canto?
Um canto é um ponto cuja vizinhança local está em duas direções de borda dominantes e diferentes. Em outras palavras, um canto pode ser interpretado como a junção de duas arestas, onde uma aresta é uma mudança repentina no brilho da imagem. Os cantos são as características importantes da imagem e geralmente são denominados pontos de interesse que são invariáveis à translação, rotação e iluminação.

[fig]


Então, vamos entender por que os cantos são considerados recursos melhores ou bons para mapeamento de patch. Na figura acima, se pegarmos a região plana, nenhuma mudança de gradiente é observada em qualquer direção. Da mesma forma, na região da borda, nenhuma mudança de gradiente é observada ao longo da direção da borda. Portanto, tanto a região plana quanto a região da borda são ruins para a correspondência de remendo, uma vez que não são muito distintas (há muitos remendos semelhantes ao longo da borda na região da borda). Enquanto na região do canto, observamos uma mudança significativa de gradiente em todas as direções. Devido a isso, os cantos são considerados bons para a correspondência de patch (mudar a janela em qualquer direção produz uma grande mudança na aparência) e geralmente mais estáveis ​​em relação à mudança do ponto de vista.

Pseudocódigo de alto nível
Pegue a escala de cinza da imagem original
2. Aplique um filtro Gaussiano para suavizar qualquer ruído
3. Aplique o operador Sobel para encontrar os valores de gradiente xey para cada pixel na imagem em tons de cinza
4. Para cada pixel p na imagem em tons de cinza, considere uma janela 3 × 3 ao redor dele e calcule a função de força do canto. Chame isso de valor Harris.
5. Encontre todos os pixels que excedem um certo limite e são os máximos locais dentro de uma determinada janela (para evitar enganos redundantes de recursos)
6. Para cada pixel que atende aos critérios em 5, calcule um descritor de recurso.

#### Passos Implementação Harris

1. Importar Bibliotecas
```python
import matplotlib.pyplot as plt
import numpy as np
import cv2
```

2. Passos iniciais quando trabalhamos com imagem
```python
# Leitura de Imagem
imagem = cv2.imread('file.jpg')
# Fazemos uma cópia da imagem
imagem_copia = np.copy(imagem)
# Mudar imagem par RGB (trabalhamos no OpenCV com BGR)
imagem_copia_RGB = cv2.cvtColor(imagem_copia, cv2.COLOR_BGR2RGB)
plt.imshow(imagem_copia_RGB)
```

3. Detecção de quinas
```python
# Convertemos a imagem para escala de cinza
cinza = cv2.cvtColor(imagem_copia_RGB, cv2.COLOR_RGB2GRAY)
# Detectamos as quinas
quinas = cv2.cornerHarris(cinza, 2, 3, 0.04)
# Dilatamos as quinas para aparecerem melhor na imagem
quinas_destacadas = cv2.dilate(quinas,None)
plt.imshow(quinas_destacadas, cmap='gray')
```

4. Agora vamos extrair as principais quinas e apresentar apenas elas
```python
# Colocaremos um parâmetro 0.2 que pode ser alterado, brinque de alterar este valor e verificar o que ocorre
through  = 0.1*quinas_destacadas.max()
# Criamos mais uma cópia, desta vez para desenhar as quinas por cima
figura_quinas = np.copy(imagem_copia_RGB)
# Iterate through all the corners and draw them on the image (if they pass the threshold)
for j in range(0, quinas_destacadas.shape[0]):
    for i in range(0, quinas_destacadas.shape[1]):
        if(quinas_destacadas[j,i] > through):
            cv2.circle( figura_quinas, (i, j), 2, (255,0,255), 1)
plt.imshow(figura_quinas)
```

### SIFT (Transformação de Característica Invariante de Escala)

SIFT significa Scale-Invariant Feature Transform e foi apresentado pela primeira vez em 2004, por D.Lowe, da University of British Columbia. SIFT é invariância à escala e rotação da imagem. Este algoritmo é patenteado, portanto, este algoritmo está incluído no módulo Non-free no OpenCV.

As principais vantagens do SIFT são
Localidade: os recursos são locais, portanto robustos à oclusão e desordem (sem segmentação anterior)
Distinção: características individuais podem ser combinadas a um grande banco de dados de objetos
Quantidade: muitos recursos podem ser gerados até mesmo para objetos pequenos
Eficiência: desempenho próximo do tempo real
Extensibilidade: pode ser facilmente estendida a uma ampla gama de diferentes tipos de recursos, com cada um adicionando robustez


SIFT é um algoritmo bastante complexo. Existem, principalmente , quatro etapas envolvidas no algoritmo SIFT. Vamos vê-los um por um.
Seleção de pico em espaço de escala: localização potencial para encontrar recursos.
Localização do ponto-chave: localizar com precisão os pontos-chave do recurso.
Atribuição de orientação: Atribuição de orientação aos pontos-chave.
Descritor de ponto-chave: Descrever os pontos-chave como um vetor de alta dimensão.
Correspondência de pontos-chave


#### Passos Implementação SIFT

1. Importar Bibliotecas
```python
import matplotlib.pyplot as plt
import numpy as np
import cv2
```

2. Passos iniciais quando trabalhamos com imagem
```python
# Leitura de Imagem
imagem = cv2.imread('file2.jpg')
# Mudar imagem par RGB (trabalhamos no OpenCV com BGR)
imagem_RGB = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
# Convertemos a imagem para escala de cinza
cinza = cv2.cvtColor(imagem_RGB, cv2.COLOR_RGB2GRAY)
plt.imshow(cinza)
```

3. Vamos começar criando nossa segunda imagem, que será usada para o teste. Para isso, iremos realizar alterações de rotação e escala. 
```python
teste = cv2.pyrDown(imagem_RGB)
teste = cv2.pyrDown(teste)
num_rows, num_cols = teste.shape[:2]
imagem_rotacao = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 1)
teste_RGB = cv2.warpAffine(teste, imagem_rotacao, (num_cols, num_rows))
teste_cinza = cv2.cvtColor(teste_RGB, cv2.COLOR_RGB2GRAY)

# Vamos mostrar a imagem na tela
fx, plots = plt.subplots(1, 2, figsize=(20,10))

plots[0].set_title("Imagem Inicial")
plots[0].imshow(imagem_RGB)

plots[1].set_title("Imagem para Teste")
plots[1].imshow(teste_RGB)
```

4. Agora vamos encontrar os pontos principais e criar nosso algoritmo
```python
sift = cv2.xfeatures2d.SIFT_create()

train_keypoints, train_descriptor = sift.detectAndCompute(training_gray, None)
test_keypoints, test_descriptor = sift.detectAndCompute(test_gray, None)
keypoints_without_size = np.copy(training_image)
keypoints_with_size = np.copy(training_image)

cv2.drawKeypoints(training_image, train_keypoints, keypoints_without_size, color = (0, 255, 0))
cv2.drawKeypoints(training_image, train_keypoints, keypoints_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display image with and without keypoints size
fx, plots = plt.subplots(1, 2, figsize=(20,10))
plots[0].set_title("Train keypoints With Size")
plots[0].imshow(keypoints_with_size, cmap='gray')
plots[1].set_title("Train keypoints Without Size")
plots[1].imshow(keypoints_without_size, cmap='gray')

# Print the number of keypoints detected in the training image
print("Number of Keypoints Detected In The Training Image: ", len(train_keypoints))

# Print the number of keypoints detected in the query image
print("Number of Keypoints Detected In The Query Image: ", len(test_keypoints))
```

5. Agora vamos realizar a parte de matching dos Keypoints
```python
# Create a Brute Force Matcher object.
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = False)

# Perform the matching between the SIFT descriptors of the training image and the test image
matches = bf.match(train_descriptor, test_descriptor)

# The matches with shorter distance are the ones we want.
matches = sorted(matches, key = lambda x : x.distance)
result = cv2.drawMatches(training_image, train_keypoints, test_gray, test_keypoints, matches, test_gray, flags = 2)

# Display the best matching points
plt.rcParams['figure.figsize'] = [14.0, 7.0]
plt.title('Best Matching Points')
plt.imshow(result)
plt.show()

# Print total number of matching points between the training and query images
print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))
```

### SURF (recurso robusto acelerado)
O método SURF (Speeded Up Robust Features) é um algoritmo rápido e robusto para representação local invariante de similaridade e comparação de imagens. O principal interesse da abordagem SURF está em seu rápido cálculo de operadores usando filtros de caixa, permitindo, assim, aplicativos em tempo real, como rastreamento e reconhecimento de objetos. A estrutura SURF descrita neste artigo é baseada no doutorado. tese de H. Bay [ETH Zurich, 2009], e mais especificamente no artigo co-escrito por H. Bay, A. Ess, T. Tuytelaars e L. Van Gool.
SURF é composto por duas etapas
Extração de característica
Descrição do Recurso

#### Passos Implementação SIFT

1. Importar Bibliotecas
```python
import matplotlib.pyplot as plt
import numpy as np
import cv2
```

2. Passos iniciais quando trabalhamos com imagem
```python
# Leitura de Imagem
imagem = cv2.imread('file3.jpg')
# Mudar imagem par RGB (trabalhamos no OpenCV com BGR)
imagem_RGB = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
# Convertemos a imagem para escala de cinza
cinza = cv2.cvtColor(imagem_RGB, cv2.COLOR_RGB2GRAY)
plt.imshow(cinza)
```

3. Vamos começar criando nossa segunda imagem, que será usada para o teste. Para isso, iremos realizar alterações de rotação e escala. 
```python
teste = cv2.pyrDown(imagem_RGB)
teste = cv2.pyrDown(teste)
num_rows, num_cols = teste.shape[:2]
imagem_rotacao = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 1)
teste_RGB = cv2.warpAffine(teste, imagem_rotacao, (num_cols, num_rows))
teste_cinza = cv2.cvtColor(teste_RGB, cv2.COLOR_RGB2GRAY)

# Vamos mostrar a imagem na tela
fx, plots = plt.subplots(1, 2, figsize=(20,10))

plots[0].set_title("Imagem Inicial")
plots[0].imshow(imagem_RGB)

plots[1].set_title("Imagem para Teste")
plots[1].imshow(teste_RGB)
```
4. Agora vamos encontrar os pontos principais e criar nosso algoritmo
```python
surf = cv2.xfeatures2d.SURF_create(800)

train_keypoints, train_descriptor = surf.detectAndCompute(training_gray, None)
test_keypoints, test_descriptor = surf.detectAndCompute(test_gray, None)

keypoints_without_size = np.copy(training_image)
keypoints_with_size = np.copy(training_image)

cv2.drawKeypoints(training_image, train_keypoints, keypoints_without_size, color = (0, 255, 0))

cv2.drawKeypoints(training_image, train_keypoints, keypoints_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display image with and without keypoints size
fx, plots = plt.subplots(1, 2, figsize=(20,10))

plots[0].set_title("Train keypoints With Size")
plots[0].imshow(keypoints_with_size, cmap='gray')

plots[1].set_title("Train keypoints Without Size")
plots[1].imshow(keypoints_without_size, cmap='gray')

# Print the number of keypoints detected in the training image
print("Number of Keypoints Detected In The Training Image: ", len(train_keypoints))

# Print the number of keypoints detected in the query image
print("Number of Keypoints Detected In The Query Image: ", len(test_keypoints))
```

5. Agora vamos realizar a parte de matching dos Keypoints
```python
# Create a Brute Force Matcher object.
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = False)

# Perform the matching between the SURF descriptors of the training image and the test image
matches = bf.match(train_descriptor, test_descriptor)

# The matches with shorter distance are the ones we want.
matches = sorted(matches, key = lambda x : x.distance)

result = cv2.drawMatches(training_image, train_keypoints, test_gray, test_keypoints, matches, test_gray, flags = 2)

# Display the best matching points
plt.rcParams['figure.figsize'] = [14.0, 7.0]
plt.title('Best Matching Points')
plt.imshow(result)
plt.show()

# Print total number of matching points between the training and query images
print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))
```


## Demais algoritmos

Além destes algoritmos descritos acima temos alguns outros muito importantes, que vale dar uma olhada em seus funcionamentos. Eles são:
* Algoritmo FAST (Featyre from Accelerated Segment Test)
* Algoritmo BRIEF (Bynary Robust Independent Elementary Features)
* Algoritmo ORB (Oriented Fast and Rotated BRIEF)


