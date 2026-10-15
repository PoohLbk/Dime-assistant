import React, { useState } from "react";
import { Link } from "react-router-dom";
import { base44 } from "@/api/base44Client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { LogIn, Mail, Lock, Loader2, ShieldCheck } from "lucide-react";
import AuthLayout from "@/components/AuthLayout";
import GoogleIcon from "@/components/GoogleIcon";
import { safeReturnTo } from "@/lib/authReturnTo";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const returnTo = safeReturnTo();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await base44.auth.loginViaEmailPassword(email, password);
      window.location.href = returnTo;
    } catch (err) {
      setError(err.message || "Invalid email or password");
    } finally {
      setLoading(false);
    }
  };

  const handleGoogle = () => {
    base44.auth.loginWithProvider("google", returnTo);
  };

  return (
    <AuthLayout
      icon={LogIn}
      title="Welcome Back"
      subtitle="Access your Emerald Ultimate Analytics dashboard"
      footer={
        <span className="text-xs text-[#7A838D]">
          Don't have an account?{" "}
          <Link
            to={"/register" + (returnTo !== "/" ? "?returnTo=" + encodeURIComponent(returnTo) : "")}
            className="text-[#2ED88A] font-bold hover:underline ml-1"
          >
            Create One
          </Link>
        </span>
      }
    >
      <div className="space-y-5">
        <Button
          variant="outline"
          className="w-full h-12 text-xs font-bold uppercase tracking-wider bg-[#0F1216] border-[#1E232A] text-[#F2F5F7] hover:bg-[#1E232A] hover:text-[#2ED88A] rounded-xl transition-all duration-200 flex items-center justify-center gap-2"
          onClick={handleGoogle}
        >
          <GoogleIcon className="w-4 h-4" />
          Continue with Google
        </Button>

        <div className="relative flex items-center justify-center my-4">
          <div className="w-full border-t border-[#1E232A]" />
          <span className="bg-[#0C0F12] px-3 text-[10px] font-bold uppercase tracking-widest text-[#7A838D] absolute">
            Or With Email
          </span>
        </div>

        {error && (
          <div className="p-3.5 rounded-xl border border-[#FF5252]/30 bg-[#FF5252]/[0.08] text-[#FF8A8A] text-xs font-medium animate-in fade-in duration-200">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
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
            <div className="flex items-center justify-between">
              <Label htmlFor="password" className="text-xs font-bold uppercase tracking-wider text-[#7A838D]">Password</Label>
              <Link to="/forgot-password" className="text-xs text-[#2ED88A] hover:underline font-semibold">
                Forgot?
              </Link>
            </div>
            <div className="relative group">
              <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-[#7A838D] group-focus-within:text-[#2ED88A] transition-colors" aria-hidden="true" />
              <Input
                id="password"
                type="password"
                autoComplete="current-password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="pl-10 h-12 bg-[#0F1216] border-[#1E232A] text-[#F2F5F7] placeholder:text-[#4A525A] focus-visible:ring-1 focus-visible:ring-[#2ED88A] focus-visible:border-[#2ED88A] rounded-xl transition-all"
                required
              />
            </div>
          </div>

          <Button 
            type="submit" 
            className="w-full h-12 font-bold tracking-wide uppercase text-xs bg-[#2ED88A] text-[#07080A] hover:bg-[#38ef9b] hover:shadow-[0_0_20px_rgba(46,216,138,0.35)] active:scale-[0.98] rounded-xl transition-all duration-200 mt-2" 
            disabled={loading}
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin" />
                Logging in...
              </span>
            ) : (
              "Log in"
            )}
          </Button>
        </form>
      </div>
    </AuthLayout>
  );
}
