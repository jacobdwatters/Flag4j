package com.flag4j.complex_numbers;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class CNumberToStringTest {
    CNumber a;
    String expStr;

    @Test
    void realToStringTestCase() {
        // ---------- Sub-case 1 ------------
        a = new CNumber(1);
        expStr = "1";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), CNumber.length(a));

        // ---------- Sub-case 2 ------------
        a = new CNumber(93.234);
        expStr = "93.234";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), CNumber.length(a));

        // ---------- Sub-case 3 ------------
        a = new CNumber(-1.23e-5);
        expStr = "-1.23E-5";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), CNumber.length(a));

        // ---------- Sub-case 4 ------------
        a = new CNumber();
        expStr = "0";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), CNumber.length(a));
    }


    @Test
    void imaginaryToStringTestCase() {
        // ---------- Sub-case 1 ------------
        a = new CNumber(0, 1);
        expStr = "i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), CNumber.length(a));

        // ---------- Sub-case 2 ------------
        a = new CNumber(0, 93.234);
        expStr = "93.234i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), CNumber.length(a));

        // ---------- Sub-case 3 ------------
        a = new CNumber(0, -1.23e-5);
        expStr = "-1.23E-5i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), CNumber.length(a));

        // ---------- Sub-case 4 ------------
        a = new CNumber(0, -1);
        expStr = "-i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), CNumber.length(a));

        // ---------- Sub-case 5 ------------
        a = new CNumber(0, 24);
        expStr = "24i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), CNumber.length(a));

        // ---------- Sub-case 6 ------------
        a = new CNumber(0, -56);
        expStr = "-56i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), CNumber.length(a));
    }


    @Test
    void complexToStringTestCase() {
        // ---------- Sub-case 1 ------------
        a = new CNumber(234.3, 1);
        expStr = "234.3 + i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), CNumber.length(a));

        // ---------- Sub-case 2 ------------
        a = new CNumber(1.341, 93.234);
        expStr = "1.341 + 93.234i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), CNumber.length(a));

        // ---------- Sub-case 3 ------------
        a = new CNumber(-9.324, -1.23e-5);
        expStr = "-9.324 - 1.23E-5i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), CNumber.length(a));

        // ---------- Sub-case 4 ------------
        a = new CNumber(994.242, -1);
        expStr = "994.242 - i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), CNumber.length(a));
    }
}
