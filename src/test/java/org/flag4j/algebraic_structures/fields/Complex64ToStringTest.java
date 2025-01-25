package org.flag4j.algebraic_structures.fields;

import org.flag4j.algebraic_structures.Complex64;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class Complex64ToStringTest {
    Complex64 a;
    String expStr;

    @Test
    void realToStringTestCase() {
        // ---------- Sub-case 1 ------------
        a = new Complex64(1);
        expStr = "1";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), Complex64.length(a));

        // ---------- Sub-case 2 ------------
        a = new Complex64(93.234f);
        expStr = "93.234";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), Complex64.length(a));

        // ---------- Sub-case 3 ------------
        a = new Complex64(-1.23e-5f);
        expStr = "-1.23E-5";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), Complex64.length(a));

        // ---------- Sub-case 4 ------------
        a = Complex64.ZERO;
        expStr = "0";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), Complex64.length(a));
    }


    @Test
    void imaginaryToStringTestCase() {
        // ---------- Sub-case 1 ------------
        a = new Complex64(0, 1);
        expStr = "i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), Complex64.length(a));

        // ---------- Sub-case 2 ------------
        a = new Complex64(0, 93.234f);
        expStr = "93.23400115966797i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), Complex64.length(a));

        // ---------- Sub-case 3 ------------
        a = new Complex64(0, -1.23e-5f);
        expStr = "-1.2299999980314169E-5i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), Complex64.length(a));

        // ---------- Sub-case 4 ------------
        a = new Complex64(0, -1);
        expStr = "-i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), Complex64.length(a));

        // ---------- Sub-case 5 ------------
        a = new Complex64(0, 24);
        expStr = "24i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), Complex64.length(a));

        // ---------- Sub-case 6 ------------
        a = new Complex64(0, -56);
        expStr = "-56i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), Complex64.length(a));
    }


    @Test
    void complexToStringTestCase() {
        // ---------- Sub-case 1 ------------
        a = new Complex64(234.3f, 1);
        expStr = "234.3 + i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), Complex64.length(a));

        // ---------- Sub-case 2 ------------
        a = new Complex64(1.341f, 93.234f);
        expStr = "1.341 + 93.23400115966797i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), Complex64.length(a));

        // ---------- Sub-case 3 ------------
        a = new Complex64(-9.324f, -1.23e-5f);
        expStr = "-9.324 - 1.2299999980314169E-5i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), Complex64.length(a));

        // ---------- Sub-case 4 ------------
        a = new Complex64(994.242f, -1);
        expStr = "994.242 - i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), Complex64.length(a));
    }
}
