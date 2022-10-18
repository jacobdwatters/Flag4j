package com.flag4j.complex_numbers;


/**
 * A lexer for producing the tokens of a complex number represented as a string.
 */
class CNumberLexer {
    private final String errMsg = "Unexpected symbol while parsing CNumber: $s";

    protected String content = ""; // Content of Lexer

    /**
     * @param content - String representation of complex number
     */
    public CNumberLexer(String content) {
        this.content = content;
    }


    /**
     * Gets the content of this Lexer.
     * @return content of Lexer
     */
    public String getContent() { return content; }


    /**
     * @param code - ascii value of character
     * @return returns true if character is digit. Otherwise, returns false.
     */
    protected boolean digit(int code ) {
        return 48<=code && code<=57;
    }



    /**
     * Produces individual symbols from content, left to right, as ascii values.
     *
     * @return Returns ascii value of the next symbol from content. If content is empty then returns -1
     */
    protected int getNextSymbol() {
        int result = -1;

        if (content.length() > 0) {
            result = content.charAt(0);
            content = content.substring(1, content.length());
        }

        return result;
    }


    /**
     * Replaces unneeded symbol back into content string.
     *
     * Note: This method should only be used when the programmer is confident the token
     * is not an unexpected token.
     *
     * @param sym - symbol to place back into content string
     */
    protected void putBackSymbol(int sym) {
        if(sym == -1) {
            content = "";
        }
        else {
            content = (char) sym + content;
        }

    }


    /**
     * Produces next CNumberToken from complex number string. Also removes this
     * CNumberToken from the string
     *
     * @return Next CNumberToken in string
     */
    public CNumberToken getNextToken() {

        int state = 1;  // state of FA
        String data = "";  // specific info for the CNumberToken
        boolean done = false;
        int sym;  // holds current symbol


        do {
            sym = getNextSymbol(); // Will return -1 if there is no symbol to get

            if(state == 1) {
                if(sym == 9 || sym == 10 || sym == 13 ||
                        sym == 32) { // Whitespace
                    state = 1;
                }
                else if(sym == 45) {
                    state = 2;
                    data += (char) sym;
                }
                else if(sym == 43) {
                    state = 4;
                    data += (char) sym;
                    done = true;
                }
                else if(digit(sym)) {
                    state = 3;
                    data += (char) sym;
                }
                else if(sym == 'i') {
                    state = 6;
                    data += (char) sym;
                    done = true;
                }
                else if(sym == -1) { // We have reached the end of the string
                    state = 5;
                    done = true;
                }
                else {
                    error(String.format(errMsg, (char) sym));
                }
            }

            else if(state == 2) {
                if(sym == 9 || sym == 10 || sym == 13 ||
                        sym == 32) { // Whitespace
                    putBackSymbol(sym);
                    done = true;
                }
                else if(digit(sym)) {
                    state = 3;
                    data += (char) sym;
                }
                else {
                    putBackSymbol(sym);
                    done = true;
                }
            }
            else if(state == 3) {
                if(digit(sym)) {
                    state = 3;
                    data += (char) sym;
                }
                else if(sym == '.') {
                    state = 7;
                    data += (char) sym;
                }
                else {
                    putBackSymbol(sym);
                    done = true;
                }
            }
            else if(state == 7) {
                if(digit(sym)) {
                    state = 8;
                    data += (char) sym;
                }
                else {
                    error(String.format(errMsg, (char) sym));
                }
            }
            else if(state == 8) {
                if(digit(sym)) {
                    state = 8;
                    data += (char) sym;
                }
                else {
                    putBackSymbol(sym);
                    done = true;
                }
            }

        } while(!done);

        if(state == 2 || state == 4) { // we have an operator
            return new CNumberToken("opp", data);
        }
        else if(state == 3 || state == 8) { // we have a number
            return new CNumberToken("num", data);
        }
        else if(state == 5) { // end of number
            return new CNumberToken("eof", data);
        }
        else if(state == 6) { // we have the imaginary unit
            return new CNumberToken("im", data);
        }
        else {
            error("somehow Lexer FA halted in bad state " + state );
            return null;
        }
    }


    /**
     * Stops execution with an error message
     * @param message - error message to print
     */
    protected static void error( String message ) {
        throw new RuntimeException(message);
    }
}
