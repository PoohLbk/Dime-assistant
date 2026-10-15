import React, { useState } from "react";
import { Link } from "react-router-dom";
import { base44 } from "@/api/base44Client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { UserPlus, Mail, Lock, Loader2, KeyRound } from "lucide-react";
import { InputOTP, InputOTPGroup, InputOTPSlot } from "@/components/ui/input-otp";
import AuthLayout from "@/components/AuthLayout";
import GoogleIcon from "@/components/GoogleIcon";
import { toast } from "@/components/ui/use-toast";
import { safeReturnTo } from "@/lib/authReturnTo";

export default function Register() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [showOtp, setShowOtp] = useState(false);
  const [otpCode, setOtpCode] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    if (password !== confirmPassword) {
      setError("Passwords do not match");
      return;
    }
    setLoading(true);
    try {
      await base44.auth.register({ email, password });
      setShowOtp(true);
    } catch (err) {
      setError(err.message || "Registration failed");
    } finally {
      setLoading(false);
    }
  };

  const handleVerify = async () => {
    setError("");
    setLoading(true);
    try {
      const result = await base44.auth.verifyOtp({ email, otpCode });
      if (result?.access_token) {
        base44.auth.setToken(result.access_token);
      }
      window.location.href = safeReturnTo();
    } catch (err) {
      setError(err.message || "Invalid verification code");
    } finally {
      setLoading(false);
    }
  };

  const handleResend = async () => {
    setError("");
    try {
      await base44.auth.resendOtp(email);
      toast({
        title: "Code sent",
        description: "Check your email for the new code.",
      });
    } catch (err) {
      setError(err.message || "Failed to resend code");
    }
  };

  const handleGoogle = () => {
    base44.auth.loginWithProvider("google", safeReturnTo());
  };

  if (showOtp) {
    return (
      <AuthLayout
        icon={KeyRound}
        title="Verify Email"
        subtitle={`We sent a 6-digit code to ${email}`}
      >
        {error && (
          <div className="mb-4 p-3.5 rounded-xl border border-[#FF5252]/30 bg-[#FF5252]/[0.08] text-[#FF8A8A] text-xs font-medium">
            {error}
          </div>
        )}
        <div className="flex justify-center mb-6">
          <InputOTP
            maxLength={6}
            value={otpCode}
            onChange={setOtpCode}
            autoFocus
            autoComplete="one-time-code"
          >
            <InputOTPGroup className="gap-2">
              {[0, 1, 2, 3, 4, 5].map((idx) => (
                <InputOTPSlot key={idx} index={idx} className="w-10 h-12 bg-[#0F1216] border-[#1E232A] text-[#2ED88A] font-bold rounded-xl text-lg focus:border-[#2ED88A]" />
              ))}
            </InputOTPGroup>
          </InputOTP>
        </div>
        <Button
          className="w-full h-12 font-bold tracking-wide uppercase text-xs bg-[#2ED88A] text-[#07080A] hover:bg-[#38ef9b] hover:shadow-[0_0_20px_rgba(46,216,138,0.35)] rounded-xl transition-all"
          onClick={handleVerify}
          disabled={loading || otpCode.length < 6}
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin" />
              Verifying...
            </span>
          ) : (
            "Verify & Continue"
          )}
        </Button>
        <p className="text-center text-xs text-[#7A838D] mt-4">
          Didn't receive code?{" "}
          <button onClick={handleResend} className="text-[#2ED88A] font-bold hover:underline">
            Resend
          </button>
        </p>
      </AuthLayout>
    );
  }

  return (
    <AuthLayout
      icon={UserPlus}
      title="Create Account"
      subtitle="Join Emerald Analytics for real-time market insights"
      footer={
        <span className="text-xs text-[#7A838D]">
          Already have an account?{" "}
          <Link
            to={"/login" + (safeReturnTo() !== "/" ? "?returnTo=" + encodeURIComponent(safeReturnTo()) : "")}
            className="text-[#2ED88A] font-bold hover:underline ml-1"
          >
            Log in
          </Link>
        </span>
      }
    >
      <div className="space-y-4">
        <Button
          variant="outline"
          className="w-full h-12 text-xs font-bold uppercase tracking-wider bg-[#0F1216] border-[#1E232A] text-[#F2F5F7] hover:bg-[#1E232A] hover:text-[#2ED88A] rounded-xl transition-all flex items-center justify-center gap-2"
          onClick={handleGoogle}
        >
          <GoogleIcon className="w-4 h-4" />
          Continue with Google
        </Button>

        <div className="relative flex items-center justify-center my-3">
          <div className="w-full border-t border-[#1E232A]" />
          <span className="bg-[#0C0F12] px-3 text-[10px] font-bold uppercase tracking-widest text-[#7A838D] absolute">
            Or Register Below
          </span>
        </div>

        {error && (
          <div className="p-3.5 rounded-xl border border-[#FF5252]/30 bg-[#FF5252]/[0.08] text-[#FF8A8A] text-xs font-medium">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-3.5">
          <div className="space-y-1.5">
            <Label htmlFor="email" className="text-xs font-bold uppercase tracking-wider text-[#7A838D]">Email</Label>
            <div className="relative group">
              <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-[#7A838D] group-focus-within:text-[#2ED88A] transition-colors" aria-hidden="true" />
              <Input
                id="email"
                type="email"
                autoComplete="email"
                autoFocus
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="pl-10 h-12 bg-[#0F1216] border-[#1E232A] text-[#F2F5F7] placeholder:text-[#4A525A] focus-visible:ring-1 focus-visible:ring-[#2ED88A] focus-visible:border-[#2ED88A] rounded-xl transition-all"
                required
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="password" className="text-xs font-bold uppercase tracking-wider text-[#7A838D]">Password</Label>
            <div className="relative group">
              <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-[#7A838D] group-focus-within:text-[#2ED88A] transition-colors" aria-hidden="true" />
              <Input
                id="password"
                type="password"
                autoComplete="new-password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="pl-10 h-12 bg-[#0F1216] border-[#1E232A] text-[#F2F5F7] placeholder:text-[#4A525A] focus-visible:ring-1 focus-visible:ring-[#2ED88A] focus-visible:border-[#2ED88A] rounded-xl transition-all"
                required
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="confirm" className="text-xs font-bold uppercase tracking-wider text-[#7A838D]">Confirm Password</Label>
            <div className="relative group">
              <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-[#7A838D] group-focus-within:text-[#2ED88A] transition-colors" aria-hidden="true" />
              <Input
                id="confirm"
                type="password"
                autoComplete="new-password"
                placeholder="••••••••"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                className="pl-10 h-12 bg-[#0F1216] border-[#1E232A] text-[#F2F5F7] placeholder:text-[#4A525A] focus-visible:ring-1 focus-visible:ring-[#2ED88A] focus-visible:border-[#2ED88A] rounded-xl transition-all"
                required
              />
            </div>
          </div>

          <Button 
            type="submit" 
            className="w-full h-12 font-bold tracking-wide uppercase text-xs bg-[#2ED88A] text-[#07080A] hover:bg-[#38ef9b] hover:shadow-[0_0_20px_rgba(46,216,138,0.35)] active:scale-[0.98] rounded-xl transition-all mt-2" 
            disabled={loading}
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin" />
                Creating...
              </span>
            ) : (
              "Create Account"
            )}
          </Button>
        </form>
      </div>
    </AuthLayout>
  );
}
