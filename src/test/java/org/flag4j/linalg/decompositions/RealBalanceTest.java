package org.flag4j.linalg.decompositions;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.linalg.decompositions.balance.RealBalancer;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class RealBalanceTest {
    static Shape aShape;
    static double[] aData;
    static Matrix a;

    static Shape permuteShape;
    static double[] permuteData;
    static Matrix permute;
    static Matrix permuteAct;

    static Shape scaleShape;
    static double[] scaleData;
    static Matrix scale;
    static Matrix scaleAct;

    static Shape permuteScaleShape;
    static double[] permuteScaleData;
    static Matrix permuteScale;
    static Matrix permuteScaleAct;

    static RealBalancer scaler;
    static RealBalancer permutor;
    static RealBalancer permutorScaler;

    static void applyBalancers() {
        permuteAct = permutor.decompose(a).getB();
        scaleAct = scaler.decompose(a).getB();
        permuteScaleAct = permutorScaler.decompose(a).getB();
    }


    @BeforeAll
    static void setUp() {
        permutor = new RealBalancer(true, false);
        scaler = new RealBalancer(false, true);
        permutorScaler = new RealBalancer(true, true);
    }


    @Test
    void testRealBalance() {
        // ----------------- sub-case 1 -----------------
        aShape = new Shape(5, 5);
        aData = new double[]{
                0.0, 0.0, 0.0, 100.2331, -140.0,
                0.0, 0.0, 1.2, 2.54, 142.0,
                0.0, 3.4, 0.0, 4.12, -10022.2212,
                0.0, 0.0, 0.0, 10.2, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0};
        a = new Matrix(aShape, aData);

        permuteShape = new Shape(5, 5);
        permuteData = new double[]{
                0.0, 3.4, 0.0, 4.12, -10022.2212,
                1.2, 0.0, 0.0, 2.54, 142.0,
                0.0, 0.0, 0.0, 100.2331, -140.0,
                0.0, 0.0, 0.0, 10.2, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0};
        permute = new Matrix(permuteShape, permuteData);

        scaleShape = new Shape(5, 5);
        scaleData = new double[]{0.0, 0.0, 0.0, 12.5291375, -140.0, 0.0, 0.0, 2.4, 0.00031005859375, 0.138671875, 0.0,
                1.7, 0.0, 0.00025146484375, -4.8936626953125, 0.0, 0.0, 0.0, 10.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        scale = new Matrix(scaleShape, scaleData);

        permuteScaleShape = new Shape(5, 5);
        permuteScaleData = new double[]{0.0, 1.7, 0.0, 2.06, -5011.1106, 2.4, 0.0, 0.0, 2.54, 142.0, 0.0, 0.0, 0.0, 100.2331,
                -140.0, 0.0, 0.0, 0.0, 10.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        permuteScale = new Matrix(permuteScaleShape, permuteScaleData);

        applyBalancers();

        assertEquals(permute, permuteAct);
        assertEquals(scale, scaleAct);
        assertEquals(permuteScale, permuteScaleAct);

        // ----------------- sub-case 2 -----------------
        aShape = new Shape(8, 8);
        aData = new double[]{6126.624252232483, 5230.904223529219, 4678.687283176287, 6395.835360234681, 5878.74954814557,
                7661.393200389235, 1826.4770608063936, 1726.6184250862443, 4122.877054785548, 1567.8469057995724,
                1744.7348734829984, 4521.044175276936, 2580.3262607639517, 5347.3621114901225, 2831.919283087424,
                6653.323956023176, 228.24242258927816, 237.69014045114713, 413.4411312454667, -7.448499599173523,
                678.4904375130443, 531.07217089675, 1012.9861688923027, 1016.378925274297, 1996.796215054365, 3739.871106883846,
                3730.836820889853, 3176.9808955419717, 986.9301184998963, 1315.6826089152178, 274.6334002238601, 508.9702755605151,
                97.04662999347228, 4135.047978558794, 267.844896567297, 112.08773771250591, 4308.866135869701, 2458.791420115115,
                3168.1532792779053, 4417.578092703147, 5062.9928116183555, 888.9483506139015, 1633.3083054064934, 1468.33492214697,
                4185.479384919863, 4687.930261898332, 3544.9689149538503, 1394.9297307361285, 622.9511113246125, 302.6582576726524,
                4511.255493925161, 1317.27667526414, 358.102095933098, 4519.853248988414, 4755.999451680448, 4168.373649093314,
                2604.2069814615215, 2041.019729854635, 866.1523849592389, 4646.076957590683, 704.2815854464203, 1084.4385900164125,
                2294.4890418962063, 6013.927731024033};
        a = new Matrix(aShape, aData);

        permuteShape = new Shape(8, 8);
        permuteData = new double[]{6126.624252232483, 5230.904223529219, 4678.687283176287, 6395.835360234681, 5878.74954814557,
                7661.393200389235, 1826.4770608063936, 1726.6184250862443, 4122.877054785548, 1567.8469057995724, 1744.7348734829984,
                4521.044175276936, 2580.3262607639517, 5347.3621114901225, 2831.919283087424, 6653.323956023176, 228.24242258927816,
                237.69014045114713, 413.4411312454667, -7.448499599173523, 678.4904375130443, 531.07217089675, 1012.9861688923027,
                1016.378925274297, 1996.796215054365, 3739.871106883846, 3730.836820889853, 3176.9808955419717, 986.9301184998963,
                1315.6826089152178, 274.6334002238601, 508.9702755605151, 97.04662999347228, 4135.047978558794, 267.844896567297,
                112.08773771250591, 4308.866135869701, 2458.791420115115, 3168.1532792779053, 4417.578092703147, 5062.9928116183555,
                888.9483506139015, 1633.3083054064934, 1468.33492214697, 4185.479384919863, 4687.930261898332, 3544.9689149538503,
                1394.9297307361285, 622.9511113246125, 302.6582576726524, 4511.255493925161, 1317.27667526414, 358.102095933098,
                4519.853248988414, 4755.999451680448, 4168.373649093314, 2604.2069814615215, 2041.019729854635, 866.1523849592389,
                4646.076957590683, 704.2815854464203, 1084.4385900164125, 2294.4890418962063, 6013.927731024033};
        permute = new Matrix(permuteShape, permuteData);

        scaleShape = new Shape(8, 8);
        scaleData = new double[]{6126.624252232483, 5230.904223529219, 2339.3436415881433, 6395.835360234681, 5878.74954814557,
                7661.393200389235, 1826.4770608063936, 1726.6184250862443, 4122.877054785548, 1567.8469057995724, 872.3674367414992,
                4521.044175276936, 2580.3262607639517, 5347.3621114901225, 2831.919283087424, 6653.323956023176, 456.4848451785563,
                475.38028090229426, 413.4411312454667, -14.896999198347046, 1356.9808750260886, 1062.1443417935, 2025.9723377846053,
                2032.757850548594, 1996.796215054365, 3739.871106883846, 1865.4184104449264, 3176.9808955419717, 986.9301184998963,
                1315.6826089152178, 274.6334002238601, 508.9702755605151, 97.04662999347228, 4135.047978558794, 133.9224482836485,
                112.08773771250591, 4308.866135869701, 2458.791420115115, 3168.1532792779053, 4417.578092703147, 5062.9928116183555,
                888.9483506139015, 816.6541527032467, 1468.33492214697, 4185.479384919863, 4687.930261898332, 3544.9689149538503,
                1394.9297307361285, 622.9511113246125, 302.6582576726524, 2255.6277469625807, 1317.27667526414, 358.102095933098,
                4519.853248988414, 4755.999451680448, 4168.373649093314, 2604.2069814615215, 2041.019729854635, 433.07619247961946,
                4646.076957590683, 704.2815854464203, 1084.4385900164125, 2294.4890418962063, 6013.927731024033};
        scale = new Matrix(scaleShape, scaleData);

        permuteScaleShape = new Shape(8, 8);
        permuteScaleData = new double[]{6126.624252232483, 5230.904223529219, 2339.3436415881433, 6395.835360234681, 5878.74954814557,
                7661.393200389235, 1826.4770608063936, 1726.6184250862443, 4122.877054785548, 1567.8469057995724, 872.3674367414992,
                4521.044175276936, 2580.3262607639517, 5347.3621114901225, 2831.919283087424, 6653.323956023176, 456.4848451785563,
                475.38028090229426, 413.4411312454667, -14.896999198347046, 1356.9808750260886, 1062.1443417935, 2025.9723377846053,
                2032.757850548594, 1996.796215054365, 3739.871106883846, 1865.4184104449264, 3176.9808955419717, 986.9301184998963,
                1315.6826089152178, 274.6334002238601, 508.9702755605151, 97.04662999347228, 4135.047978558794, 133.9224482836485,
                112.08773771250591, 4308.866135869701, 2458.791420115115, 3168.1532792779053, 4417.578092703147, 5062.9928116183555,
                888.9483506139015, 816.6541527032467, 1468.33492214697, 4185.479384919863, 4687.930261898332, 3544.9689149538503,
                1394.9297307361285, 622.9511113246125, 302.6582576726524, 2255.6277469625807, 1317.27667526414, 358.102095933098,
                4519.853248988414, 4755.999451680448, 4168.373649093314, 2604.2069814615215, 2041.019729854635, 433.07619247961946,
                4646.076957590683, 704.2815854464203, 1084.4385900164125, 2294.4890418962063, 6013.927731024033};
        permuteScale = new Matrix(permuteScaleShape, permuteScaleData);

        applyBalancers();

        assertEquals(permute, permuteAct);
        assertEquals(scale, scaleAct);
        assertEquals(permuteScale, permuteScaleAct);

        // ----------------- sub-case 3 -----------------
        aShape = new Shape(11, 11);
        aData = new double[]{1e-08, 0.02, 0.0, 5e-05, 30000.0, 0.0, -1000.0, 0.0, 70.0, 0.0, 9000000.0, 5000.0, 100.0, -0.002, 0.0,
                0.0, 0.0, 9e-09, 0.0, 0.0, -300000.0, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-09,
                100000000.0, -10.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, -0.0003, 40000.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -400.0, 0.0, 10.0,
                200.0, 0.0, 200000.0, 0.0, 1e-05, 0.0, 0.0, 0.0, 0.0, 5e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2000000000.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.005, 0.0002, 1e-07, 0.0, 0.0, 10.0, 0.0, -90000.0, 0.0, 3e-06, 0.0,
                0.0, 0.0, 300000000.0, 0.0, 0.0, 0.0, 0.0005, 0.0, 0.0, 10000.0, 0.0, 0.0, 0.0, 0.0, 1e-08, 200000.0, 7000.0, 0.0,
                0.0, 0.2, 0.0, 0.0, 1000.0, 0.0, 0.0, -400.0, 0.01};
        a = new Matrix(aShape, aData);

        permuteShape = new Shape(11, 11);
        permuteData = new double[]{0.0, 0.0, 5e-06, 0.0, 1e-05, 200.0, 0.0, 0.0, 0.0, 200000.0, 0.0, 0.0, 100.0, -300000.0, 0.0, 0.0,
                5000.0, 200.0, 0.0, 0.0, -0.002, 9e-09, 0.0, 0.0005, 1e-08, 0.0, 10000.0, 0.0, 200000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 100000000.0, -10.0, 0.0, 0.0, 40.0, 0.0, 1e-09, 0.0, 0.0, 40000.0, 0.0, 1.0, 0.0, -0.0003, 10.0, 0.0, -400.0,
                0.0, 0.0, 0.0, 0.02, 0.0, 5e-05, 30000.0, 1e-08, 9000000.0, 0.0, 70.0, 0.0, -1000.0, 0.0, 0.0, -400.0, 0.2, 0.0,
                7000.0, 0.01, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0002, 1e-07, 100.0, 0.005, 0.0, 0.0, 0.0,
                0.0, 3e-06, 10.0, 0.0, 0.0, 300000000.0, -90000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2000000000.0};
        permute = new Matrix(permuteShape, permuteData);

        scaleShape = new Shape(11, 11);
        scaleData = new double[]{1e-08, 0.005, 0.0, 1.5625e-06, 7500.0, 0.0, -31.25, 0.0, 2.1875, 0.0, 281250.0, 20000.0, 100.0,
                -0.00025, 0.0, 0.0, 0.0, 1.125e-09, 0.0, 0.0, -75000.0, 25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1e-09, 100000000.0, -80.0, 0.0, 0.0, 80.0, 0.0, 0.0, 0.0, -0.0012, 40000.0, 0.0, 0.125, 0.0, 0.0, 0.0, 0.0,
                -50.0, 0.0, 1.25, 6400.0, 0.0, 200000.0, 0.0, 8e-05, 0.0, 0.0, 0.0, 0.0, 1e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                2000000000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0025, 0.0002, 5e-08, 0.0, 0.0, 320.0, 0.0,
                -90000.0, 0.0, 2.4e-05, 0.0, 0.0, 0.0, 300000000.0, 0.0, 0.0, 0.0, 0.002, 0.0, 0.0, 40000.0, 0.0, 0.0, 0.0, 0.0,
                1e-08, 100000.0, 224000.0, 0.0, 0.0, 0.2, 0.0, 0.0, 1000.0, 0.0, 0.0, -800.0, 0.01};
        scale = new Matrix(scaleShape, scaleData);

        permuteScaleShape = new Shape(11, 11);
        permuteScaleData = new double[]{0.0, 0.0, 1e-05, 0.0, 8e-05, 6400.0, 0.0, 0.0, 0.0, 200000.0, 0.0, 0.0, 100.0, -75000.0, 0.0,
                0.0, 20000.0, 25.0, 0.0, 0.0, -0.00025, 1.125e-09, 0.0, 0.002, 1e-08, 0.0, 40000.0, 0.0, 100000.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 100000000.0, -80.0, 0.0, 0.0, 0.00244140625, 0.0, 1e-09, 0.0, 0.0, 40000.0, 0.0, 0.125, 0.0, -0.0012,
                1.25, 0.0, -50.0, 0.0, 0.0, 0.0, 0.005, 0.0, 1.5625e-06, 7500.0, 1e-08, 281250.0, 0.0, 2.1875, 0.0, -31.25, 0.0, 0.0,
                -800.0, 0.2, 0.0, 224000.0, 0.01, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0002, 0.0016384,
                1638400.0, 81.92, 0.0, 0.0, 0.0, 0.0, 2.4e-05, 320.0, 0.0, 0.0, 300000000.0, -90000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2000000000.0};
        permuteScale = new Matrix(permuteScaleShape, permuteScaleData);

        applyBalancers();

        assertEquals(permute, permuteAct);
        assertEquals(scale, scaleAct);
        assertEquals(permuteScale, permuteScaleAct);

        // ----------------- sub-case 4 -----------------
        aShape = new Shape(8, 8);
        aData = new double[]{3.0, 1.0, 0.0, 1.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0, 1.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0};
        a = new Matrix(aShape, aData);

        permuteShape = new Shape(8, 8);
        permuteData = new double[]{3.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0,
                1.0, 1.0, 0.0, 0.0, 5.0, 0.0, 0.0, 3.0, 1.0, 0.0, 4.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0};
        permute = new Matrix(permuteShape, permuteData);

        scaleShape = new Shape(8, 8);
        scaleData = new double[]{3.0, 1.0, 0.0, 2.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 2.0, 1.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0};
        scale = new Matrix(scaleShape, scaleData);

        permuteScaleShape = new Shape(8, 8);
        permuteScaleData = new double[]{3.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                3.0, 1.0, 1.0, 0.0, 0.0, 5.0, 0.0, 0.0, 3.0, 1.0, 0.0, 4.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0};
        permuteScale = new Matrix(permuteScaleShape, permuteScaleData);

        applyBalancers();

        assertEquals(permute, permuteAct);
        assertEquals(scale, scaleAct);
        assertEquals(permuteScale, permuteScaleAct);
    }
}
