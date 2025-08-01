# Быстрый старт
Amvera стремиться сделать доставку кода и развертывания приложения максимально простым и понятным. Код своего приложения
можно загружать двумя способами:

1. Используя git репозиторий. Подробно о работе с git и использованием репозиториев написано в [соответствующем разделе](git.rst).
2. Через интерфейс. Всегда можно загрузить нужные файлы в удобном графическом окне.

## Процесс создания приложения

### Инициализация

1. Перейдите в [личный кабинет Amvera](https://cloud.amvera.ru) при необходимости авторизовавшись. 
2. Для открытия формы создания приложения необходимо нажать кнопку "Создать".
3. В появившемся окне требуется указать название для вашего приложения, выбрать тип сервиса "Приложение", необходимый тарифный план и нажать "Далее".
   ![quick-start1](../img/quick-start1.png)
4. После окончания индикатора загрузки приложение будет создано и откроется следующий этап создания проекта "Загрузка данных".

### Загрузка данных
Для хранения пользовательского кода вне зависимости от метода загрузки используется [git](git.rst) репозиторий. При использовании загрузки через
интерфейс всегда можно переключиться на доставку кода через git и наоборот. 

#### Загрузка через git

Для загрузки данных приложения с использованием git потребуется:

1. **Привязать к Amvera свой репозиторий или склонировать созданный в Amvera.**
   Информация о том, как это сделать находится в разделе [git](git.rst) этой документации.

   **Предположим**, что на локальном компьютере ещё нет git репозитория. Выполним команду, указанную первой на текущей странице.
     ```shell
     git clone https://git.amvera.ru/<username>/<service-slug>
     ```
     ![quick-start2](../img/quick-start2.png)
   Теперь на локальном компьютере есть папка, связанная с удалённым репозиторием Amvera. В эту папку нужно скопировать код
   проекта.

2. **Создать файл конфигурации.**

   Подробно о том, что такое файл конфигурации и зачем он нужен можно найти в [соответсвующем разделе](configuration/config-file.md).
   - Сгенерировать файл конфигурации через [генератор yaml файла](https://manifest.amvera.ru/).
   - Воспользоваться следующим шагом создания проекта (потребуется сделать `git pull`).
   - В некоторых случаях может потребоваться написать файл самостоятельно. Инструкцию для вашего окружения можно найти на 
[странице поддерживаемых окружений](supported-env.rst).
   - Если нужно воспользоваться сборкой и запуском Dockerfile, то это описано [тут](configuration/docker.md).
     ```{eval-rst}
     .. admonition:: Важно
        :class: warning
    
        Сгенерированный файл конфигурации нужно положить в корневую папку репозитория.
     ```
3. **Создать файл с зависимостями.**

   Прописать корректно все зависимости в файл, чтобы наше облако смогло их установить.
   ```{eval-rst}
   .. admonition:: Важно
      :class: warning

      Для Python необходимо наличие файла requirements.txt с требуемыми библиотеками.
   ```

4. **Сделать push в master ветку.**

    Если все шаги выполнены правильно, ваш проект [автоматически соберется](build.md) и [развернется](run.md) после отправки кода в master.

#### Загрузка через интерфейс

Для развертывания через интерфейс необходимо загрузить файлы проекта, в том числе файл с необходимыми зависимостями.

```{eval-rst}
.. admonition:: Важно
   :class: warning

   Для Python необходимо наличие файла requirements.txt с требуемыми библиотеками.
```

После того как файлы были перетянуты нужно дождаться их появления списка загруженных файлов и проверить, что ничего не 
потерялось.
![quick-start2](../img/quick-start4.png)

### Задание конфигурации
[Конфигурацию](configuration/config-file.md) можно задать используя последний этап при создании приложения. Требуется выбрать используемое вашим приложением
окружение и инструмент. Чтобы лучше ознакомится с вводимыми параметрами следует ознакомиться с необходимым окружением в 
[данном разделе](supported-env.rst).

![quick-start2](../img/quick-start5.png)
```{eval-rst}
.. admonition:: Важно
   :class: warning

   Поскольку файл конфигурации хранится в том же репозитории, что и код, то его добавление создает новый commit в репозиторий Amvera.
   
   Если используется git необходимо будет сделать pull, чтобы забрать его локально.
```
После нажатия кнопки "Завершить" (при условии загрузки файлов через интерфейс) начнется сборка и впоследствии запуск приложения.

### Примеры

Примеры развертывания проектов для разных стеков находятся в разделе ["Примеры"](../general/examples.rst).

### Если вы столкнулись с ошибкой или вам что-то непонятно

- Ознакомьтесь с разделом [Частые ошибки](../general/faq.rst). Мы постарались собрать самые частые ошибки, возникающие у наших пользователей.
- Пишите в поддержку на почту support@amvera.ru. Мы обязательно ответим и постараемся вам помочь. Обычно, ответ занимает от 5 минут до суток.
- Если ничего не помогает, вы можете написать нашему CEO, Кириллу Косолапову на почту kkosolapov@amvera.ru. Отвечает он дольше чем поддержка, но обязательно с вами свяжется и поможет решить ваш вопрос.