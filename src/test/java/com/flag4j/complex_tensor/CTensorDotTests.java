package com.flag4j.complex_tensor;

import com.flag4j.CTensor;
import com.flag4j.Shape;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CTensorDotTests {

    static int[] aAxes;
    static int[] bAxes;

    static CNumber[] aEntries;
    static CNumber[] bEntries;
    static CNumber[] expEntries;

    static Shape aShape;
    static Shape bShape;
    static Shape expShape;

    static CTensor A;
    static CTensor B;
    static CTensor exp;

    @BeforeEach
    void setup() {
        aEntries = new CNumber[]{
                new CNumber("1.4415+9.14i"), new CNumber("235.61-9.865i"), new CNumber("-0.00024+5.15i"),
                new CNumber("1.0"), new CNumber("-0.0-85.1i"), new CNumber("1.345+3.5i"),
                new CNumber("0.014+0.01i"), new CNumber("-140.0-0.0235i"), new CNumber("1.5+9.24i"),
                new CNumber("51.0"), new CNumber("6.1-265.55i"), new CNumber("-0.00014+4.14i")};
        bEntries = new CNumber[]{
                new CNumber("13.41+3.4i"), new CNumber("-99.23"), new CNumber("14.0+0.094i"), new CNumber("0.000245-9.0i"),
                new CNumber("0.0+1.25i"), new CNumber("95.14"), new CNumber("0.0+546.0i"), new CNumber("15.6+9234.6i"),
                new CNumber("1.45656"), new CNumber("0.009345"), new CNumber("156.21326-0.00351i"), new CNumber("125.6"),
                new CNumber("144.5+40.1i"), new CNumber("-1545.4+2.0i"), new CNumber("1.145-8.25i"), new CNumber("7.4"),
                new CNumber("-9.345"), new CNumber("0.0"), new CNumber("314.5"), new CNumber("0.0+4.155i"),
                new CNumber("-0.0"), new CNumber("145.5"), new CNumber("7.0"), new CNumber("1.4456+103.525i")
        };

        aShape = new Shape(3, 2, 1, 2);
        bShape = new Shape(4, 3, 2);

        A = new CTensor(aShape, aEntries);
        B = new CTensor(bShape, bEntries);
    }


    @Test
    void multiAxesTestCase() {
        // --------------------- Sub-case 1 ---------------------
        aAxes = new int[]{0};
        bAxes = new int[]{1};
        expEntries = new CNumber[]{
                new CNumber("-15.296085 - 1062.0565i"), new CNumber("-766.2300449999999 - 27.88944950000007i"), new CNumber("-4756.0876776000005 + 2106.5110014i"),
                new CNumber("-84193.35660000001 + 14614.0086405i"), new CNumber("-874.30975 + 1194.74685i"), new CNumber("-2245.9741000000004 - 14751.813000000002i"),
                new CNumber("463.85175 + 2939.21i"), new CNumber("-992.3793000000001 - 12207.415723499998i"), new CNumber("3211.572100000001 + 781.6607800000002i"),
                new CNumber("-18495.939970475003 + 966.7998075i"), new CNumber("13355.125333199998 + 128647.97895i"), new CNumber("101180.45756902502 + 2175610.2447075i"),
                new CNumber("33995.051525 + 8015.379750000001i"), new CNumber("-364082.01100000006 + 15742.491i"), new CNumber("74456.345 - 3102.5425i"),
                new CNumber("310.412175 + 6767.984550000001i"), new CNumber("314.6193416 + 76.82700000000001i"), new CNumber("580.4678186299999 - 25775.58749755i"),
                new CNumber("-1859.9108026600004 - 41482.569078399996i"), new CNumber("-46792.03361317 - 33274.95621055i"), new CNumber("-263.45565 + 3225.6260760000005i"),
                new CNumber("-9.825504 - 7958.736480000001i"), new CNumber("42.62452 - 239.17499999999995i"), new CNumber("27480.52066 + 249.07742280000008i"),
                new CNumber("-1951.7627909999999 - 10.089175000000001i"), new CNumber("-99.48911960000001 + 1653.8795942425i"), new CNumber("-203.9257384564 + 1192.6886677314i"),
                new CNumber("14.274116000000001 + 9754.5837803925i"), new CNumber("-15.992566700000003 + 1156.3847924999998i"), new CNumber("-2581.4 + 1.8261i"),
                new CNumber("314.49902 + 28.979999999999997i"), new CNumber("-20798.593702384 + 6.706040499999999i")};
        expShape = new Shape(2, 1, 2, 4, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.tensorDot(B, aAxes, bAxes));

        // --------------------- Sub-case 2 ---------------------
        aAxes = new int[]{0, 1};
        bAxes = new int[]{1, 2};
        expEntries = new CNumber[]{
                new CNumber("565.17173363 - 26837.64399755i"), new CNumber("-51548.12129077 - 31168.445209150002i"), new CNumber("-884.1352539999999 - 6763.989630000002i"),
                new CNumber("27944.37241 + 3188.2874228i"), new CNumber("3112.0829804000005 + 2435.5403742425i"), new CNumber("13369.3994492 + 138402.56273039253i"),
                new CNumber("31413.651525 + 8017.205850000001i"), new CNumber("53657.751297616 - 3095.8364595i")
        };
        expShape = new Shape(1, 2, 4);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.tensorDot(B, aAxes, bAxes));

        // --------------------- Sub-case 3 ---------------------
        aAxes = new int[]{1};
        bAxes = new int[]{0};

        assertThrows(LinearAlgebraException.class, ()->A.tensorDot(B, aAxes, bAxes));

        // --------------------- Sub-case 4 ---------------------
        aAxes = new int[]{0, 1};
        bAxes = new int[]{1};

        assertThrows(IllegalArgumentException.class, ()->A.tensorDot(B, aAxes, bAxes));

        // --------------------- Sub-case 5 ---------------------
        aAxes = new int[]{0, 1};
        bAxes = new int[]{1, 0};

        assertThrows(LinearAlgebraException.class, ()->A.tensorDot(B, aAxes, bAxes));

        // --------------------- Sub-case 5 ---------------------
        aAxes = new int[]{0, 1, 2, 3};
        bAxes = new int[]{1, 0, 3, 2};

        assertThrows(IllegalArgumentException.class, ()->A.tensorDot(B, aAxes, bAxes));
    }


    @Test
    void singleAxesTestCase() {
        // --------------------- Sub-case 1 ---------------------
        expEntries = new CNumber[]{
                new CNumber("-15.296085 - 1062.0565i"), new CNumber("-766.2300449999999 - 27.88944950000007i"), new CNumber("-4756.0876776000005 + 2106.5110014i"),
                new CNumber("-84193.35660000001 + 14614.0086405i"), new CNumber("-874.30975 + 1194.74685i"), new CNumber("-2245.9741000000004 - 14751.813000000002i"),
                new CNumber("463.85175 + 2939.21i"), new CNumber("-992.3793000000001 - 12207.415723499998i"), new CNumber("3211.572100000001 + 781.6607800000002i"),
                new CNumber("-18495.939970475003 + 966.7998075i"), new CNumber("13355.125333199998 + 128647.97895i"), new CNumber("101180.45756902502 + 2175610.2447075i"),
                new CNumber("33995.051525 + 8015.379750000001i"), new CNumber("-364082.01100000006 + 15742.491i"), new CNumber("74456.345 - 3102.5425i"),
                new CNumber("310.412175 + 6767.984550000001i"), new CNumber("314.6193416 + 76.82700000000001i"), new CNumber("580.4678186299999 - 25775.58749755i"),
                new CNumber("-1859.9108026600004 - 41482.569078399996i"), new CNumber("-46792.03361317 - 33274.95621055i"), new CNumber("-263.45565 + 3225.6260760000005i"),
                new CNumber("-9.825504 - 7958.736480000001i"), new CNumber("42.62452 - 239.17499999999995i"), new CNumber("27480.52066 + 249.07742280000008i"),
                new CNumber("-1951.7627909999999 - 10.089175000000001i"), new CNumber("-99.48911960000001 + 1653.8795942425i"), new CNumber("-203.9257384564 + 1192.6886677314i"),
                new CNumber("14.274116000000001 + 9754.5837803925i"), new CNumber("-15.992566700000003 + 1156.3847924999998i"), new CNumber("-2581.4 + 1.8261i"),
                new CNumber("314.49902 + 28.979999999999997i"), new CNumber("-20798.593702384 + 6.706040499999999i")};
        expShape = new Shape(2, 1, 2, 4, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.tensorDot(B, 0, 1));

        // --------------------- Sub-case 1 ---------------------
        assertThrows(LinearAlgebraException.class, ()->A.tensorDot(B, 1, 1));
    }


    @Test
    void noAxesTestCase() {
        // --------------------- Sub-case 1 ---------------------
        A = A.reshape(2, 2, 3);
        expEntries = new CNumber[]{
                new CNumber("3281.284325 + 11.505539999999993i"), new CNumber("-231.79015415 - 2537.483616925i"), new CNumber("-4647.2793130824 + 1577.1883254424001i"),
                new CNumber("-84379.58496855001 + 14101.007711575001i"), new CNumber("30.1721928 - 624.6705250000001i"), new CNumber("-502.46010000000024 - 14195.074000000002i"),
                new CNumber("453.35006999999996 + 2910.5800000000004i"), new CNumber("33710.124203056 - 1421.9480735i"), new CNumber("17.034399999999998 - 1186.3187499999997i"),
                new CNumber("-737.1667 + 332.9691505i"), new CNumber("210.11911969999997 + 968.78843305i"), new CNumber("184.53199999999998 + 9673.4047405i"),
                new CNumber("-570.1440249999999 - 90.047i"), new CNumber("-1545.4 - 627.74i"), new CNumber("323.915 + 24.5i"),
                new CNumber("-360.39316800000006 - 12233.594275i"), new CNumber("-1971.394051 - 11.432300000000001i"), new CNumber("141.07498 + 2138.1012942425i"),
                new CNumber("24.973922399999964 + 1451.01502824i"), new CNumber("94.96409999999997 + 1289.9841803924999i"), new CNumber("-172.889375 + 1070.6316924999999i"),
                new CNumber("-1057.6556 - 15.5999i"), new CNumber("14.903 + 67.825i"), new CNumber("-21324.444150000003 + 165.28376400000005i"),
                new CNumber("789.0967 - 3543.726775i"), new CNumber("-7450.691825100001 + 338.91454024999996i"), new CNumber("8.8776775436 + 28105.933388891397i"),
                new CNumber("795.6394204999999 + 471482.10243525i"), new CNumber("5185.6983083000005 + 1652.03195i"), new CNumber("-78770.26000000001 - 1863.0700000000002i"),
                new CNumber("16039.49902 + 28.979999999999997i"), new CNumber("458.956297616 - 38419.6497095i")};
        expShape = new Shape(2, 2, 4, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.tensorDot(B));

        // --------------------- Sub-case 1 ---------------------
        A = A.reshape(2, 2, 3);
        B = B.reshape(4, 2, 3);
        assertThrows(IllegalArgumentException.class, ()->A.tensorDot(B));
    }
}
