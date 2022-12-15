/*
 * MIT License
 *
 * Copyright (c) 2022 Jacob Watters
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

package com.flag4j.complex_numbers;

/**
 * A parser for complex numbers represented as a string.
 */
public class CNumberParser {

    /**
     * Parses a complex number in the form of a string into its real and imaginary parts.
     * For example, the string <code>"2+3i"</code> would be parsed into real and imaginary parts
     * <code>2</code> and <code>3</code> respectivly
     *
     * @param num - complex number in one of three forms: <code>a + bi, a,</code> or <code>bi</code> where a and b are
     * 				real numbers and i is the imaginary unit sqrt(-1)
     * @return The complex number represented by the string num.
     */
    public static CNumber parseNumber(String num) {
        double[] result = new double[2];

        CNumberLexer lex = new CNumberLexer(num);

        CNumberToken token;
        CNumberToken operator;
        CNumberToken real;
        CNumberToken imaginary;

        token = lex.getNextToken();
        if(token.matches("im", "i")) { // then we have the imaginary unit
            result[0] = 0;
            result[1] = 1;
        }
        else {
            real = token;
            token = lex.getNextToken();

            if(token.matches("eof", "")) { // Then we have a real number (a)
                result[0] = Double.parseDouble(real.getDetails());
                result[1] = 0;
            }
            else if(token.matches("im", "i")) { // Then we have a pure imaginary number (bi)
                imaginary = real;
                result[0] = 0;

                token = lex.getNextToken();
                token.errorCheck("eof", "");

                if(imaginary.getDetails().matches("-")) {
                    result[1] = -1;
                }
                else {
                    result[1] = Double.parseDouble(imaginary.getDetails());
                }
            }
            else { // Then we have a complex number with nonzero real and imaginary parts (a + bi)
                operator = token;

                if(!operator.isKind("opp") && operator.isKind("num")) {
                    imaginary = operator;
                }
                else {
                    imaginary = lex.getNextToken();
                }

                if(imaginary.matches("im", "i")) { // Then we have the unit imaginary number
                    token = lex.getNextToken();
                    token.errorCheck("eof", "");

                    result[0] = Double.parseDouble(real.getDetails());
                    result[1] = 1;

                    if(operator.getDetails().equals("-")) { // The operator is negative
                        result[1] = -result[1];
                    }
                }
                else { // Then we have a multiple of the unit imaginary number
                    imaginary.errorCheck("num");

                    token = lex.getNextToken();
                    token.errorCheck("im", "i");

                    token = lex.getNextToken();
                    token.errorCheck("eof", "");

                    result[0] = Double.parseDouble(real.getDetails());
                    result[1] = Double.parseDouble(imaginary.getDetails());

                    if(operator.getDetails().equals("-")) { // The operator is negative
                        result[1] = -result[1];
                    }
                }
            }
        }

        return new CNumber(result[0], result[1]);
    }
}
