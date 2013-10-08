.process <- function(df){
    # Read in the table
    # Name the cols
    colnames(df) <- c("roi", "sub", "set", "acc")
    print(str(df))

    # Drop overall
    mask <- df[["set"]] != "overall"
    df <- df[mask,]
    df
}


.create.control.factor <- function(df){
    # Create a label for the control factor,
    # name it roi_type
    control_mask <- (df[["roi"]] == "left_ventrical") | (df[["roi"]] == "right_ventrical")
    control <- rep("exp", nrow(df))
    control[control_mask] = "control"

    df[["roi_type"]] <- control
    df
}


plot.acctable.hist <- function(df, name, height, width){
    library("ggplot2")
    
    df <- .process(df) 
    df <- .create.control.factor(df)

    # Get median of control
    control_mask <- df[["roi_type"]] == "control"
    control_median <- median(df[control_mask,"acc"])

    # And plot.
    pdf(file=name, height=height, width=width)

    p <- ggplot(aes(x=acc, fill=roi_type), data=df)
    p <- p + facet_grid(roi~.)
    p <- p + geom_density(aes(alpha=0.5)) + theme_bw()
    p <- p + geom_vline(xintercept = control_median, colour="grey")

    print(p)
    dev.off()

}


plot.acctable.box <- function(df, name, height, width){
    library("ggplot2")
    
    df <- .process(df)
    df <- .create.control.factor(df)

    # Get median of control
    control_mask <- df[["roi_type"]] == "control"
    control_median <- median(df[control_mask,"acc"])

    # And plot.
    pdf(file=name, height=height, width=width)

    p <- ggplot(aes(x=roi, y=acc, fill=roi_type), data=df)
    p <- p + geom_boxplot(notch=TRUE) + theme_bw()
    p <- p + geom_hline(yintercept = control_median, 
            colour="grey") + coord_flip()

    print(p)
    dev.off()
}


