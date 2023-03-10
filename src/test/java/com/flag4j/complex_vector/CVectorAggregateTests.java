package com.flag4j.complex_vector;

import com.flag4j.CVector;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CVectorAggregateTests {

    static CNumber[] aEntries;
    static CVector a;
    CNumber expComplex;
    int[] expIndices;

    @BeforeAll
    static void setup() {
        aEntries = new CNumber[]{new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
        new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257)};
        a = new CVector(aEntries);
    }


    @Test
    void sumTest() {
        // ------------------ Sub-case 1 ------------------
        expComplex = aEntries[0].add(aEntries[1]).add(aEntries[2]).add(aEntries[3]).add(aEntries[4]);
        assertEquals(expComplex, a.sum());
    }


    @Test
    void minTest() {
        // ------------------ Sub-case 1 ------------------
        expComplex = new CNumber(0.009257);
        assertEquals(expComplex, a.min());
        assertEquals(expComplex, a.minAbs());
    }


    @Test
    void maxTest() {
        // ------------------ Sub-case 1 ------------------
        expComplex = new CNumber(1.455, 6126.347).mag();
        assertEquals(expComplex, a.max());
        assertEquals(expComplex, a.maxAbs());
    }


    @Test
    void argMinTest() {
        // ------------------ Sub-case 1 ------------------
        expIndices = new int[]{4};
        assertArrayEquals(expIndices, a.argMin());
    }


    @Test
    void argMaxTest() {
        // ------------------ Sub-case 1 ------------------
        expIndices = new int[]{0};
        assertArrayEquals(expIndices, a.argMax());
    }
}
