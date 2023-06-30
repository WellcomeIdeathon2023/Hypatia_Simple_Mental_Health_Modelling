
#install.packages(c('ggplot2','dplyr', 'shiny'))
library(shiny)
library(dplyr)
library(ggplot2)
library(tidyr)
library(patchwork)
library(shinythemes)
library(bslib)

source("RLmodel_fitting.R")

# Define UI for application that draws a histogram
ui <- fluidPage(

    theme = shinytheme(theme = 'cosmo'),

    tabsetPanel(


# Simulate data -----------------------------------------------------------

      tabPanel('Simulate data',
        # Application title
        titlePanel(h4(HTML(paste(
                      "This is an introduction to creating a basic algorithm that learns the value of its environment.",
                      "<br/>", "<br/>",
                      "Here the agent is placed in an environment where it learns about the value of two cards.
                      On each trial it samples a card to learn whether it gained a reward.")))),
        titlePanel(h4(HTML(paste("Move the sliders to change the task structure and agent policy.",
                  "Click 'Select a new agent' to start a new agent from scratch using the same settings",
                  sep="<br/>")))),
        br(),

        sidebarLayout(
            sidebarPanel(
              class="pull-left",
              downloadButton('downloadData', 'Download Data')
            ),

        mainPanel(
          )
        ),

        br(),

        # Sidebar with a slider input for number of bins
        sidebarLayout(
            sidebarPanel(
                actionButton("setseed", "Select a new agent"),
                br(),
                br(),
                sliderInput("lr",
                            HTML(paste("Learning Rate: (&lambda;)")),
                            min = 0.01,
                            max = 1,
                            value = 0.1,
                            step = 0.01),
                br(),
                sliderInput("tau",
                            HTML(paste("Decision Temperature: (&tau;)")),
                            min = 0.1,
                            max = 10,
                            value = 1,
                            step = 0.1),
                br(),
                sliderInput("trials",
                            "Task Length:",
                            min = 50,
                            max = 500,
                            value = 100,
                            step = 50),
                br(),
                sliderInput("winprob",
                            "Win Probability of Card 2:",
                            min = 0,
                            max = 1,
                            value = 0.8,
                            step = 0.1),
               br()
            ),

            # Show a plot of the generated distribution
            mainPanel(
               plotOutput("distPlot", height = '700px', width = 'auto')
            )
        )
        ),

# Fit own data ------------------------------------------------------------

        tabPanel("Fit your own data",
                 titlePanel(h4("Upload a .csv file of your behavioural data.
                                It needs at least one column called 'choice' and another called 'reward':")),
                  sidebarLayout(
                  sidebarPanel(
                    fileInput("file", "Choose CSV File",
                              accept = c(
                                "text/csv",
                                "text/comma-separated-values,text/plain",
                                ".csv")),

                    tags$hr(),

                    actionButton("optimize", "Optimize Parameters")
                  ),

                  mainPanel(
                    verbatimTextOutput("optimized_params"),
                    verbatimTextOutput("min_log_likelihood")
                  )
                ),
        ), ### WORK IN PROGRESS ###

# Maths -------------------------------------------------------------------

    tabPanel(title = 'The Maths',
    titlePanel(h4(HTML(paste("The way in which the agent acts upon the environment to maximise its return is to convert its beliefs into actions",
                  "<br/>",
                  "<i><center> Beliefs </i> &#8594; <i> Actions </i></center>",
                  "<br/>",
                  "The rate at which the agent will explore its environment, generate new beliefs, and make optimised actions is governed by two parameters:
                  a learning rate (how quickly an agent learns from each reward) and a decision temperature (how noisily an agent chooses
                  between each option)",sep="<br/>")))),
    titlePanel(h4("This agent uses the following equations:")),
    br(),
    withMathJax(),
      tabPanel(
      title = "Diagnostics",
      h4(textOutput("diagTitle")),
      uiOutput("formula")
      ),

      ),
    ),

    titlePanel(h5(HTML(paste("CC <a href='https://www.joebarnby.com/'>Team Hypatia</a> 2023"))))

)

# Define server logic required to draw a histogram
server <- function(input, output) {


# Maths output for maths tab ----------------------------------------------

    output$formula <- renderUI({
    withMathJax(paste0("
                       $$Q^{t}_{c} = Q^{t-1}_{c} * \\lambda + ({Reward - Q^{t-1}_{c}})$$
                       $$p(\\hat{c} = c) = \\frac{e^{\\frac{Q^{t}_{c}}{\\tau}}}{\\sum_{c'\\in(c_1, c_2)} e^{\\frac{Q^{t}_{c'}}{\\tau}}}$$

                       $$\\text{Therefore, } Q^{t}_{c} = \\text{the internal beliefs the agent holds about the value of each card at each trial}$$
                       "))
    })


# Uploaded data reactive element ------------------------------------------

    uploaded_data <- reactive({
                      file <- input$file
                      if(is.null(file)) return(NULL)
                      read.csv(file$datapath) %>%
                        na.omit()
                    })

# For simulating data -----------------------------------------------------

    data <- reactive({

        # generate bins based on input$bins from ui.R
        trials  <- input$trials
        lambda  <- input$lr
        tau     <- input$tau
        seed    <- input$setseed

        actions <- 2
        Q2      <- matrix(NA, trials+1, actions)

        Q2[1,]  <- c(0.5, 0.5) # Initialize the first two actions as equal probabilities
        R2      <- matrix(c(1-input$winprob, input$winprob, input$winprob, 1-input$winprob), 2, 2)
        a       <- r <- prob_a1 <- rep(NA, trials+1)

        #Sample the cards
        observeEvent(input$setseed, {
          set.seed(sample(1:100000, 1))
        })

        for (t in 1:trials){

          #sample an action
          a1            <- exp(Q2[t,1]/tau)
          a2            <- exp(Q2[t,2]/tau)
          prob_a1[t]    <- a1/(a1+a2)
          a[t]          <- sample(c(1,2),  1, T, prob = c(prob_a1[t], 1-prob_a1[t]))

          #sample a reward for the action
          prob_r        <- R2[a[t],]
          r[t]          <- sample(c(1, 0), 1, T, prob = prob_r)

          #update
          PE            <- r[t] - Q2[t, a[t]]

          Q2[t+1, a[t]] <- Q2[t, a[t]] + (lambda * PE) #RW equation
          Q2[t+1,-a[t]] <- Q2[t,-a[t]]

        }

       list(
          Q2 = Q2,
          prob_a1 = prob_a1,
          choice=a,
          reward=r
        )

    })


# Plot the simulated data -------------------------------------------------

    output$distPlot <- renderPlot({

        trials  <- input$trials
        seed    <- input$setseed

        #Set colours
        defcols <- c("#E41A1C" ,"#377EB8")

        #extract df
        Q2      <- data()$Q2
        prob_a1 <- data()$prob_a1
        a       <- data()$choice
        r       <- data()$reward

        colnames(Q2) <- c('Card 1', 'Card 2')
        mainplot <- Q2 %>%
          as.data.frame() %>%
          mutate(Trial = 0:trials,
                 ProbA1= prob_a1) %>%
          pivot_longer(1:2, 'Option', values_to = 'Q') %>%

          ggplot(aes(Trial, Q, color = Option))+
          geom_line()+
          geom_line(aes(Trial, ProbA1), linetype = 2, color = 'black')+
          geom_hline(yintercept = c(1-input$winprob, input$winprob),
                     color = defcols,
                     linetype = 2, alpha = 0.2)+
          coord_cartesian(ylim = c(0,1))+
          scale_color_brewer(palette = 'Set1')+
          labs(y = expression(paste('Q'[c]^t, '    &    p(' ,hat(c), '= c)')))+
          scale_y_continuous(breaks = seq(0, 1, 0.2), labels = seq(0, 1, 0.2))+
          theme_bw() +
          theme(text = element_text(size = 20),
                axis.title = element_text(size = 20),
                axis.text = element_text(size = 20),
                legend.title = element_blank(),
                legend.text = element_text(size = 20),
                legend.position = c(0.9, 0.1),
                legend.background = element_rect(color = 'black'))

        subplot <- data.frame(Action = a, Reward = r) %>%
          na.omit() %>%
          mutate(Action1  = ifelse(Action == 1, 1, 0),
                 Action2  = ifelse(Action == 2, 1, 0),
                 ActionS1 = sum(Action1),
                 ActionS2 = sum(Action2),
                 RewardS  = sum(Reward)) %>%
          dplyr::select(ActionS1, ActionS2, RewardS) %>%
          rename(`Card 1 Choices` = 1, `Card 2 Choices` = 2, `Rewards`= 3) %>%
          distinct() %>%
          pivot_longer(1:3, 'Index', values_to = 'Value') %>%
          ggplot(aes(Index, Value))+
          geom_col(fill = c(defcols, "#FFB302"), color = 'black')+
          coord_cartesian(ylim = c(0, trials))+
          labs(title = 'Sum of...')+
          theme_bw() +
          theme(text = element_text(size = 20),
                plot.title = element_text(face = 'bold', vjust = -6, hjust = 0.05),
                axis.title = element_blank(),
                axis.text = element_text(size = 20))
#
        (mainplot / subplot) & plot_layout(nrow = 2, heights = c(2,1))
    })


# Push the download data button to server ---------------------------------

    # Downloadable csv of data file
    output$downloadData <- downloadHandler(
      filename = function() {
        paste("data-", Sys.Date(), ".csv", sep="")
      },
      content = function(file) {
        write.csv(data(), file, row.names = FALSE)
      }
    )


# Push the fitting output to server ---------------------------------------

    # Render the data in the UI from uploaded data
    optimized_result <- reactiveVal()

    observeEvent(input$optimize, {
      if (is.null(uploaded_data())) return(NULL)
      data_to_use <- uploaded_data()
      start_params <- c(1, 0.1) # Initial parameters (decision temperature and learning rate)
      opt_result <- optim(start_params, fn = neg_log_likelihood, data = data_to_use,
                          upper = c(10, 1), lower = c(0, 0))
      optimized_result(opt_result)
    })

    output$optimized_params <- renderPrint({
      if (is.null(optimized_result())) return(NULL)
      optimized_result()$par
    })

    output$min_log_likelihood <- renderPrint({
      if (is.null(optimized_result())) return(NULL)
      optimized_result()$value
    })
}

# Run the application
shinyApp(ui = ui, server = server)
