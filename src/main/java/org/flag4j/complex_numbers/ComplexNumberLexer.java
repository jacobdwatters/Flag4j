/*
 * MIT License
 *
 * Copyright (c) 2022-2024. Jacob Watters
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package org.flag4j.complex_numbers;


import org.flag4j.util.exceptions.ComplexNumberParseingException;

/**
 * A lexer for producing the tokens of a complex number represented as a string.
 */
class ComplexNumberLexer {
    /**
     * Error message for unexpected symbol.
     */
    private static final String ERR_MSG = "Unexpected symbol while parsing CNumber: %s";

    /**
     * Content of the lexer.
     */
    protected String content;

    /**
     * @param content - String representation of complex number
     */
    public ComplexNumberLexer(String content) {
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
            content = content.substring(1);
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
     * Produces next {@link ComplexNumberToken} from complex number string. Also removes this
     * ComplexNumberToken from the string. This method implements a finite automata which describes the legal arrangement of
     * tokens within a complex number.
     *
     * @return Next {@link ComplexNumberToken} in string.
     * @throws RuntimeException If the string is not a valid representation of a complex number.
     */
    public ComplexNumberToken getNextToken() {
        int state = 1;  // State of Finite Automata
        boolean done = false;
        StringBuilder dataBuilder = new StringBuilder(); // specific info for the ComplexNumberToken
        int sym;  // holds current symbol


        do {
            sym = getNextSymbol(); // Will return -1 if there is no symbol to get

            if(state == 1) {
                if(sym == 45) {
                    state = 2;
                    dataBuilder.append((char) sym);
                }
                else if(sym == 43) {
                    state = 4;
                    dataBuilder.append((char) sym);
                    done = true;
                }
                else if(digit(sym)) {
                    state = 3;
                    dataBuilder.append((char) sym);
                }
                else if(sym == 'i' || sym == 'j') {
                    state = 6;
                    dataBuilder.append((char) sym);
                    done = true;
                }
                else if(sym == -1) { // We have reached the end of the string
                    state = 5;
                    done = true;
                }
                else if(!(sym == 9 || sym == 10 || sym == 13 ||
                        sym == 32)){
                    // Otherwise, if it is not a whitespace character, we have encountered an unexpected symbol.
                    error(String.format(ERR_MSG, (char) sym));
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
                    dataBuilder.append((char) sym);
                }
                else {
                    putBackSymbol(sym);
                    done = true;
                }
            }
            else if(state == 3) {
                if(digit(sym)) {
                    // State does not need to change here.
                    dataBuilder.append((char) sym);
                }
                else if(sym == '.') {
                    state = 7;
                    dataBuilder.append((char) sym);
                }
                else {
                    putBackSymbol(sym);
                    done = true;
                }
            }
            else if(state == 7) {
                if(digit(sym)) {
                    state = 8;
                    dataBuilder.append((char) sym);
                }
                else {
                    error(String.format(ERR_MSG, (char) sym));
                }
            }
            else if(state == 8) {
                if(digit(sym)) {
                    // State does not need to change here.
                    dataBuilder.append((char) sym);
                }
                else {
                    putBackSymbol(sym);
                    done = true;
                }
            }

        } while(!done);

        if(state == 2 || state == 4) { // we have an operator
            return new ComplexNumberToken("opp", dataBuilder.toString());
        }
        else if(state == 3 || state == 8) { // we have a number
            return new ComplexNumberToken("num", dataBuilder.toString());
        }
        else if(state == 5) { // end of number
            return new ComplexNumberToken("eof", dataBuilder.toString());
        }
        else if(state == 6) { // we have the imaginary unit
            return new ComplexNumberToken("im","i");
        }
        else {
            error("Somehow Lexer FA halted in bad state " + state);
            return null;
        }
    }


    /**
     * Stops execution with an error message
     * @param message - error message to print
     */
    protected static void error( String message ) {
        throw new ComplexNumberParseingException(message);
    }
}
