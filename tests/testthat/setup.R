# Tests that exercise base-graphics plotting (dendrograms, diagnostic
# panels) draw to the default device, which writes an Rplots.pdf into
# tests/testthat/ and would then be picked up by R CMD build. Send those
# draws to a null device instead. The device is closed when the test
# session exits.
grDevices::pdf(NULL)
