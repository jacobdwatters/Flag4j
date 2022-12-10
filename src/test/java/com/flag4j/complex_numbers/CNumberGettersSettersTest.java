package com.flag4j.complex_numbers;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class CNumberGettersSettersTest {
    CNumber num;
    long expReLong, expImLong;
    int expReInt, expImInt;
    float expReFloat, expImFloat;
    double expReDouble, expImDouble;


    @Test
    void valueGettersTestCase() {
        num = new CNumber(692.13, -9673.134);
        expReLong = (long) 692.13;
        expImLong = (long) -9673.134;
        expReInt = (int) 692.13;
        expImInt = (int) -9673.134;
        expReFloat = (float) 692.13;
        expImFloat = (float) -9673.134;
        expReDouble = 692.13;
        expImDouble = -9673.134;

        Assertions.assertEquals(expReLong, num.longValue());
        Assertions.assertEquals(expImLong, num.longImaginaryValue());
        Assertions.assertEquals(expReInt, num.intValue());
        Assertions.assertEquals(expImInt, num.intImaginaryValue());
        Assertions.assertEquals(expReFloat, num.floatValue());
        Assertions.assertEquals(expImFloat, num.floatImaginaryValue());
        Assertions.assertEquals(expReDouble, num.doubleValue());
        Assertions.assertEquals(expImDouble, num.doubleImaginaryValue());
    }


    @Test
    void gettersTestCase() {
        num = new CNumber(692.13, -9673.134);
        expReDouble = 692.13;
        expImDouble = -9673.134;

        Assertions.assertEquals(expReDouble, num.getReal());
        Assertions.assertEquals(expImDouble, num.getImaginary());
    }
}
