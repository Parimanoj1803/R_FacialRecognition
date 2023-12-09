library(shiny)
library(keras)
library(imager)



# Load the pre-trained model
loaded_model <- load_model_hdf5("C:\\Users\\Lenovo\\Desktop\\r setup\\model.h5")

# Define the emotions vector
EMOTIONS <- c('Angry', 'Disgust', 'Happy', 'Sad', 'Surprise', 'Neutral')

ui <- fluidPage(
  tags$head(
    tags$style(
      HTML(" @keyframes moveText {
        from {
          margin-right: 100%;  /* Start off-screen to the right */
        }
        to {
          margin-right: 0%;  /* Move to the left */
        }
      }
        body {
          background: url('https://i.pinimg.com/564x/a1/11/d9/a111d9687162ff5d7a368e92502f5343.jpg') center center;
          background-size: cover;
        }
        .title {
          text-align: center;
          font-size: 34px;
          font-weight: bold;
          color: white;
          font-family: cursive;
          text-indent: 30px;  /* Add indentation */
          border-radius: 30px;  /* Add curvature */
        }
        .other {
          text-align: center;
          font-size: 20px;
          font-weight: bold;
          color: white;
        }
        .center-image {
          display: block;
          margin-left: auto;
          margin-right: auto;
        }
        .additional-image {
          float: right;
          margin-top: 80px;
          margin-right: 20px;
          border-radius: 10px;
          box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
          transition: transform 0.3s ease-in-out;
        }
        .additional-image:hover {
          transform: scale(1.05);
        }
      ")
    )
  ),
  div(
    class = "title",
    "Facial Expression Prediction"
  ),
  div(class = "other",  tags$img(
    src = "https://i.pinimg.com/564x/14/d6/f8/14d6f81841fff58f7899e16f7f7745eb.jpg",  # Replace with the actual path to your second additional image
    width = 400,
    height = 400,
    class = "additional-image"
  ),fileInput("image", "Choose an image file")),
  div(class = "center-image", imageOutput("prediction_image")),
  div(class = "other", textOutput("prediction_text")),
  div(
    class = "floating-text",
    style = "float: right; margin-top: 80px; margin-right: 20px; color: #ffffff; font-size: 18px; font-weight: bold; animation: moveText 5s linear infinite;",
    "Know your Expression!!"
  )
  
 
)

# Define the Shiny server


# Define the Shiny server
server <- function(input, output) {
  # Read and preprocess the selected image
  selected_image <- reactive({
    req(input$image)
    img_path <- input$image$datapath
    img <- load.image(img_path)
    resized_img <- resize(img, 48, 48)
    gray_img <- grayscale(resized_img) / 255.0
    reshaped_gray <- array_reshape(gray_img, c(1,48, 48, 1))
    reshaped_gray
  })
  
  # Perform prediction when the image is selected
  output$prediction_image <- renderImage({
    img <- selected_image()[[1,25 ,25 , 1]]
    # Save the image as a temporary JPEG file
    img_path <- tempfile(fileext = ".jpg")
    imager::save.image(imager::as.cimg(img), img_path)
    
    list(
      src = img_path,
      contentType = "image/jpeg",
      width = 300,
      height = 300
    )
  },deleteFile = TRUE)
  
  #deleteFile = TRUE
  output$prediction_text <- renderText({
    prediction <- predict(loaded_model, selected_image())
    label_index <- which.max(prediction)
    label <- EMOTIONS[label_index + 1]
    
    paste("Predicted Expression: ", label)
  })
}

# Run the Shiny app
shinyApp(ui = ui, server = server)

