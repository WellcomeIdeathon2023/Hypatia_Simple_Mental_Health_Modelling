
#install.packages(c('ggplot2','dplyr', 'shiny'))
library(shiny)
library(dplyr)
library(ggplot2)
library(tidyr)
library(patchwork)
library(shinythemes)
library(bslib)
library(rstan)
library(renv)

#ran this to show the shiny app where to source it from
#options(timeout=99999)
#devtools::install_github('WellcomeIdeathon2023/Hypatia_Simple_Mental_Health_Modelling',subdir='/code/shinyStanModels')

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

        #sidebarLayout(
        #    ,
#
        #mainPanel(
        #  )
        #),

        br(),

        # Sidebar with a slider input for number of bins
        sidebarLayout(
            sidebarPanel(
                actionButton("setseed", "Select a new agent"),
                br(),
                br(),
                sliderInput("lr",
                            HTML(paste("Learning Rate: (&alpha;)")),
                            min = 0.01,
                            max = 1,
                            value = 0.1,
                            step = 0.01),
                br(),
                sliderInput("beta",
                            HTML(paste("Decision Temperature: (&beta;)")),
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
                br(),
                downloadButton('downloadData', 'Download Data')
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
                                It needs at least one column called 'choice' and another called 'reward.
                                If you have more than one participant, please add a column called 'id':")),
                  sidebarLayout(
                  sidebarPanel(
                    fileInput("file", "Choose CSV File",
                              accept = c(
                                "text/csv",
                                "text/comma-separated-values,text/plain",
                                ".csv")),

                    tags$hr(),

                    sliderInput("n_chain",
                                    HTML(paste("Number of chains")),
                                    min = 0,
                                    max = 4,
                                    value = 2,
                                    step = 1),
                      br(),
                      sliderInput("n_samp",
                                    HTML(paste("Number of samples")),
                                    min = 500,
                                    max = 2000,
                                    value = 500,
                                    step = 10),

                    actionButton("optimize", "Optimize Parameters")
                  ),

                  mainPanel(
                    h3('Optimized parameters'),
                    h6('Note: the square brackets refer to each participant id.'),
                    verbatimTextOutput("optimized_params"),
                    h3('Minimum log likelihood per participant'),
                    h6('The closer this is to zero, the better.'),
                    verbatimTextOutput("min_log_likelihood"),
                    h3('Rhat value for each optimized parameter'),
                    h6('This should be below 1.01, ideally.'),
                    verbatimTextOutput("rhat"),
                    h3('Traceplot for log posterior'),
                    h6('This should look like lots of furry caterpillars in different colours.'),
                    plotOutput("traceplot", height = '200px', width = 'auto')
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

    hr(),
    p("CC Team Hypatia 2023", style = "text-align: center;")

)

# Define server logic required to draw a histogram
server <- function(input, output) {


# Maths output for maths tab ----------------------------------------------

    output$formula <- renderUI({
    withMathJax(paste0("
                       $$Q^{t}_{c} = Q^{t-1}_{c} * \\alpha ({Reward - Q^{t-1}_{c}})$$
                       $$p(\\hat{c} = c) = \\frac{e^{\\frac{Q^{t}_{c}}{\\beta}}}{\\sum_{c'\\in(c_1, c_2)} e^{\\frac{Q^{t}_{c'}}{\\beta}}}$$

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

        seed <- sample(1:100000, 1)
        set.seed(seed)

        # generate bins based on input$bins from ui.R
        trials  <- input$trials
        alpha   <- input$lr
        beta    <- input$beta
        seed    <- input$setseed

        actions <- 2
        Q2      <- matrix(NA, trials+1, actions)

        Q2[1,]  <- c(0.5, 0.5) # Initialize the first two actions as equal probabilities
        R2      <- matrix(c(1-input$winprob, input$winprob, input$winprob, 1-input$winprob), 2, 2)
        a       <- r <- prob_a1 <- rep(NA, trials+1)

        #Sample the cards
        observeEvent(input$setseed, {
          seed <- sample(1:100000, 1)
          set.seed(seed)
        })

        for (t in 1:trials){

          #sample an action
          a1            <- exp(Q2[t,1]/beta)
          a2            <- exp(Q2[t,2]/beta)
          prob_a1[t]    <- a1/(a1+a2)
          a[t]          <- sample(c(1,2),  1, T, prob = c(prob_a1[t], 1-prob_a1[t]))

          #sample a reward for the action
          prob_r        <- R2[a[t],]
          r[t]          <- sample(c(1, 0), 1, T, prob = prob_r)

          #update
          PE            <- r[t] - Q2[t, a[t]]

          Q2[t+1, a[t]] <- Q2[t, a[t]] + (alpha * PE) #RW equation
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
          pivot_longer(cols = 1:2, names_to = 'Option', values_to = 'Q') %>%

          ggplot(aes(Trial, Q, color = Option))+
          geom_line(size = 1.1)+
          geom_line(aes(Trial, ProbA1, linetype = 'p(Action)'),color = 'black', size = 1)+
          geom_hline(yintercept = c(1-input$winprob, input$winprob),
                     color = defcols,
                     linetype = 2, alpha = 0.2)+
          coord_cartesian(ylim = c(0,1))+
          scale_color_brewer(palette = 'Set1', name = 'Q Values')+
          scale_linetype_manual(name = '', values = 2)+
          labs(title = 'Trial-by-Trial Outcomes')+
          scale_y_continuous(breaks = seq(0, 1, 0.2), labels = seq(0, 1, 0.2), expand = c(0,0))+
          scale_x_continuous(expand = c(0,0))+
          theme_bw() +
          theme(text = element_text(size = 20, family="Helvetica-Narrow"),
                axis.title = element_text(size = 20, face = 'bold', family="Helvetica-Narrow"),
                axis.text = element_text(size = 20, family="Helvetica-Narrow"),
                axis.title.y = element_blank(),
                legend.box = "horizontal",
                legend.position = "bottom",
                #legend.title = element_blank(),
                legend.text = element_text(size = 20, face = 'bold', family="Helvetica-Narrow"),
                #legend.position = c(0.15, 0.1),
                legend.background = element_blank())

        subplot <- data.frame(Action = a, Reward = r) %>%
          na.omit() %>%
          mutate(Action1  = ifelse(Action == 1, 1, 0),
                 Action2  = ifelse(Action == 2, 1, 0),
                 ActionS1 = sum(Action1),
                 ActionS2 = sum(Action2),
                 RewardS  = sum(Reward)) %>%
          dplyr::select(ActionS1, ActionS2, RewardS) %>%
          rename(`Card 1` = 1, `Card 2` = 2, `Reward`= 3) %>%
          distinct() %>%
          pivot_longer(cols = 1:3, names_to = 'Index', values_to = 'Value') %>%
          ggplot(aes(Index, Value))+
          geom_col(fill = c(defcols, "#FFB302"), color = 'black')+
          coord_cartesian(ylim = c(0, trials))+
          scale_y_continuous(expand = c(0,0))+
          labs(title = 'Average Sum of...')+
          theme_bw() +
          theme(text = element_text(size = 20, family="Helvetica-Narrow"),
                #plot.title = element_text(face = 'bold', vjust = -6, hjust = 0.05),
                axis.title = element_blank(),
                axis.text = element_text(size = 20))
#
        (mainplot / subplot) & plot_layout(nrow = 2, heights = c(2,1))
    })


# Push the download data button to server ---------------------------------

    # Downloadable csv of data file
    output$downloadData <- downloadHandler(
      filename = function() {
        paste("data-", Sys.Date(), "-", seed, ".csv", sep="")
      },
      content = function(file) {
        write.csv(data(), file, row.names = FALSE)
      }
    )


# Push the fitting output to server ---------------------------------------



    # Render the data in the UI from uploaded data
    optimized_result <- reactiveVal()

    # Load the RDS model file (precompiled)
    model <- readRDS('model_1a1b.rds')

    #optimise the parameters for the inputted data
    observeEvent(input$optimize, {
      if (is.null(uploaded_data())) return(NULL)
      data_to_use <- uploaded_data()
      #start_params <- c(1, 0.1) # Initial parameters (decision temperature and learning rate)

      ### TO NOTE ###
      ### We want to replace the below with a hierarchical fit ### ?hBayesDM use?

      #opt_result2 <- optim(start_params, fn = neg_log_likelihood, data = data_to_use,
      #                    upper = c(10, 1), lower = c(0, 0), method = 'L-BFGS-B')
      #optimized_result(opt_result2)

      data<-data_to_use

      if(is.null(data$id)){
        data$id<-rep(1,nrow(data))
      }

      if(is.null(data$trial)){
        nsub<-length(unique(data$id))
        trials<-nrow(data)/nsub #specifies
        data$trial<-rep(seq(1:trials),nsub)
      }

      reward<-data%>%
        select(id,trial,reward)%>%
        pivot_wider(id_cols = trial,names_from = id,values_from = reward)%>%
        select(-c(trial))

      choice<-data%>%
        select(id,trial,choice)%>%
        pivot_wider(id_cols = trial,names_from = id,values_from = choice)%>%
        select(-c(trial))

      data<-list(
        nsub=length(unique(data$id)),
        ntrials=max(data$trial),
        reward=reward,
        choices=choice)

      opt_result2 <- rstan::sampling(object = shinyStanModels:::stanmodels[["fit_1a1b"]],data,chains=input$n_chain,iter=input$n_samp)
      parameters<-summary(opt_result2, pars=c('alpha','beta'))$summary[,1]
      loglik<-summary(opt_result2, pars=c('loglik'))$summary[,1]
      rhat<-summary(opt_result2, pars=c('alpha','beta'))$summary[,10] #rhat of just param estimates
      trace<-traceplot(opt_result2,pars='lp__')
      fit_summary<-list(pars=parameters,loglik=loglik,rhat=rhat,traceplot=trace)
      optimized_result(fit_summary)
    })

    output$optimized_params <- renderPrint({
      if (is.null(optimized_result())) return(NULL)
      optimized_result()$par
    })

    output$min_log_likelihood <- renderPrint({
      if (is.null(optimized_result())) return(NULL)
      optimized_result()$loglik
    })

    output$rhat <- renderPrint({
      if (is.null(optimized_result())) return(NULL)
      optimized_result()$rhat
    })

    output$traceplot <- renderPlot({
      if (is.null(optimized_result())) return(NULL)
      optimized_result()$trace
    })
}

# Run the application
shinyApp(ui = ui, server = server)
